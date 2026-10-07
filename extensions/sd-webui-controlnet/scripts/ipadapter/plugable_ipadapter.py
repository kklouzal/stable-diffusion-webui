import itertools
import torch
import math
from threading import RLock
from typing import Union, Dict, Optional

from .pulid_attn import PuLIDAttnSetting
from .ipadapter_model import ImageEmbed, IPAdapterModel
from ..enums import StableDiffusionVersion, TransformerID


def get_block(model, flag):
    return {
        "input": model.input_blocks,
        "middle": [model.middle_block],
        "output": model.output_blocks,
    }[flag]


def attn_forward_hacked(self, x, context=None, **kwargs):
    batch_size, sequence_length, inner_dim = x.shape
    h = self.heads
    head_dim = inner_dim // h

    if context is None:
        context = x

    q = self.to_q(x)
    k = self.to_k(context)
    v = self.to_v(context)

    del context

    q, k, v = map(
        lambda t: t.view(batch_size, -1, h, head_dim).transpose(1, 2),
        (q, k, v),
    )

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(1, 2).reshape(batch_size, -1, h * head_dim)

    del k, v, x

    # Each hack sees the attention output so far (PuLID projects its id attention against it) and returns
    # the term to add, or None when inactive at this layer/step.
    for f in self.ipadapter_hacks:
        ip_out = f(self, out, q)
        if ip_out is not None:
            out = out + ip_out

    del q

    return self.to_out(out)


all_hacks = {}
current_model = None
_all_hacks_lock = RLock()


def hack_blk(block, function, type):
    with _all_hacks_lock:
        if not hasattr(block, "ipadapter_hacks"):
            block.ipadapter_hacks = []

        if len(block.ipadapter_hacks) == 0:
            all_hacks[block] = block.forward
            block.forward = attn_forward_hacked.__get__(block, type)
            block._controlnet_ipadapter_owner = all_hacks

        block.ipadapter_hacks.append(function)
    return


def set_model_attn2_replace(
    model,
    target_cls,
    function,
    transformer_id: TransformerID,
):
    block = get_block(model, transformer_id.block_type.value)
    module = (
        block[transformer_id.block_id][1]
        .transformer_blocks[transformer_id.block_index]
        .attn2
    )
    hack_blk(module, function, target_cls)


def clear_all_ip_adapter():
    global all_hacks, current_model
    with _all_hacks_lock:
        owned, all_hacks = all_hacks, {}
        current_model = None
        for k, v in owned.items():
            if getattr(k, "_controlnet_ipadapter_owner", None) is owned:
                k.forward = v
                k.ipadapter_hacks = []
                del k._controlnet_ipadapter_owner
    return


class PlugableIPAdapter(torch.nn.Module):
    def __init__(self, ipadapter: IPAdapterModel):
        super().__init__()
        self.ipadapter: IPAdapterModel = ipadapter
        self.disable_memory_management = True
        self.dtype = None
        self.weight: Union[float, Dict[int, float]] = 1.0
        self.cache = None
        self.image_emb = None
        self.p_start = 0.0
        self.p_end = 1.0
        self.latent_width: int = 0
        self.latent_height: int = 0
        self.effective_region_mask = None
        self.pulid_attn_setting: Optional[PuLIDAttnSetting] = None
        self.region_masks = {}
        self.cond_rows_memo = None

    def reset(self):
        self.cache = {}
        self.region_masks = {}
        self.cond_rows_memo = None

    def release_request_state(self):
        self.reset()
        self.image_emb = None
        self.effective_region_mask = None
        self.latent_width = self.latent_height = 0
        self.pulid_attn_setting = None

    @torch.no_grad()
    def hook(
        self,
        model,
        preprocessor_outputs,
        weight,
        start: float,
        end: float,
        latent_width: int,
        latent_height: int,
        effective_region_mask: Optional[torch.Tensor],
        pulid_attn_setting: Optional[PuLIDAttnSetting] = None,
        dtype=torch.float32,
    ):
        global current_model
        current_model = model

        self.p_start = start
        self.p_end = end
        self.latent_width = latent_width
        self.latent_height = latent_height
        self.effective_region_mask = effective_region_mask
        self.pulid_attn_setting = pulid_attn_setting

        self.reset()

        self.weight = weight
        device = torch.device("cpu")
        self.dtype = dtype

        self.ipadapter.to(device, dtype=self.dtype)
        if isinstance(preprocessor_outputs, (list, tuple)):
            preprocessor_outputs = preprocessor_outputs
        else:
            preprocessor_outputs = [preprocessor_outputs]
        self.image_emb = ImageEmbed.average_of(
            *[self.ipadapter.get_image_emb(o) for o in preprocessor_outputs]
        )

        if self.ipadapter.is_sdxl:
            sd_version = StableDiffusionVersion.SDXL
            from sgm.modules.attention import CrossAttention
        else:
            sd_version = StableDiffusionVersion.SD1x
            from ldm.modules.attention import CrossAttention

        input_ids, output_ids, middle_ids = sd_version.transformer_ids
        for i, transformer_id in enumerate(
            itertools.chain(input_ids, output_ids, middle_ids)
        ):
            set_model_attn2_replace(
                model,
                CrossAttention,
                self.patch_forward(i, transformer_id.transformer_index),
                transformer_id,
            )

    def weight_on_transformer(self, transformer_index: int) -> float:
        if isinstance(self.weight, dict):
            return self.weight.get(transformer_index, 0.0)
        else:
            assert isinstance(self.weight, (float, int))
            return self.weight

    def cond_rows(self, cond_mark: torch.Tensor) -> torch.Tensor:
        """(B, 1, 1) bool on cond_mark's device: True for the call's cond rows.

        cond_mark is the (B, 1, 1, 1) row mark hook.py sets per UNet call (1 for cond rows, 0 for uncond rows).
        Every attn2 layer of a call sees the same cond_mark object, so the mask is built once per call.
        """
        memo = self.cond_rows_memo
        if memo is None or memo[0] is not cond_mark:
            memo = self.cond_rows_memo = (cond_mark, cond_mark[:, :, :, 0] > 0.5)
        return memo[1]

    def call_ip(self, key: str, cond_mark: torch.Tensor, device, dtype) -> torch.Tensor:
        """`key`'s projection (to_k_ip/to_v_ip) of the image embeds for the rows of this call, (B, T, C) in `dtype`.

        The projection is row-wise and bias-free, so the cond and uncond embeds (with PuLID's zero tokens) are
        projected once per request, in the IP-Adapter's dtype on its device, and each call selects its rows by
        cond_mark. The row layout must not be frozen at the first call: A1111 runs cond and uncond rows as
        separate UNet calls (prompt and negative prompt of different token lengths without padding,
        batch_cond_uncond off) and drops the uncond rows on skipped-uncond steps.
        """
        cache_key = (key, device, dtype)
        projected = self.cache.get(cache_key)
        if projected is None:
            cond_emb, uncond_emb = self.image_emb
            emb = torch.cat([cond_emb, uncond_emb])
            if self.ipadapter.is_pulid:
                emb = self.pulid_attn_setting.append_zero_tokens(emb)
            both = self.ipadapter.ip_layers.to_kvs[key](emb).to(device=device, dtype=dtype)
            projected = self.cache[cache_key] = (both[: cond_emb.shape[0]], both[cond_emb.shape[0]:])
        cond, uncond = projected
        return torch.where(self.cond_rows(cond_mark), cond, uncond)

    def apply_effective_region_mask(self, out: torch.Tensor) -> torch.Tensor:
        if self.effective_region_mask is None:
            return out

        _, sequence_length, _ = out.shape
        mask = self.region_masks.get((sequence_length, out.device))
        if mask is None:
            # sequence_length = mask_h * mask_w
            # sequence_length = (latent_height * factor) * (latent_height * factor)
            # sequence_length = (latent_height * latent_height) * factor ^ 2
            factor = math.sqrt(sequence_length / (self.latent_width * self.latent_height))
            assert (
                factor > 0
            ), f"{factor}, {sequence_length}, {self.latent_width}, {self.latent_height}"
            mask_h = int(self.latent_height * factor)
            mask_w = int(self.latent_width * factor)

            # (1, mask_h * mask_w, 1): broadcast over the call's rows and channels.
            mask = torch.nn.functional.interpolate(
                self.effective_region_mask.to(out.device),
                size=(mask_h, mask_w),
                mode="bilinear",
            ).view(1, -1, 1)
            self.region_masks[(sequence_length, out.device)] = mask
        return out * mask

    def attn_eval(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        ip_k: torch.Tensor,
        ip_v: torch.Tensor,
        attn_heads: int,
        head_dim: int,
    ):
        """hidden_states: the attention output (B, L, heads * head_dim); query: (B, heads, L, head_dim);
        ip_k/ip_v: the call's image k/v rows (B, T, heads * head_dim) in the query's dtype."""
        if self.ipadapter.is_pulid:
            assert self.pulid_attn_setting is not None
            return self.pulid_attn_setting.eval(
                hidden_states,
                query,
                ip_k,
                ip_v,
                attn_heads,
                head_dim,
            )
        else:
            return self._attn_eval_ipadapter(
                hidden_states,
                query,
                ip_k,
                ip_v,
                attn_heads,
                head_dim,
            )

    def _attn_eval_ipadapter(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        ip_k: torch.Tensor,
        ip_v: torch.Tensor,
        attn_heads: int,
        head_dim: int,
    ):
        assert hidden_states.ndim == 3
        batch_size, sequence_length, inner_dim = hidden_states.shape

        ip_k, ip_v = map(
            lambda t: t.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2),
            (ip_k, ip_v),
        )

        ip_out = torch.nn.functional.scaled_dot_product_attention(
            query, ip_k, ip_v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        ip_out = ip_out.transpose(1, 2).reshape(batch_size, -1, attn_heads * head_dim)
        return ip_out

    @torch.no_grad()
    def patch_forward(self, number: int, transformer_index: int):
        k_key = f"{number * 2 + 1}_to_k_ip"
        v_key = f"{number * 2 + 1}_to_v_ip"

        @torch.no_grad()
        def forward(attn_blk, out, q):
            weight = self.weight_on_transformer(transformer_index)

            current_sampling_percent = getattr(
                current_model, "current_sampling_percent", 0.5
            )
            if (
                current_sampling_percent < self.p_start
                or current_sampling_percent > self.p_end
                or weight == 0.0
            ):
                return None

            batch_size, sequence_length, inner_dim = out.shape
            h = attn_blk.heads
            head_dim = inner_dim // h
            cond_mark = current_model.cond_mark
            # In the query's dtype (the dtype of the attention's own k/v, and what PuLID projects in).
            ip_out = self.attn_eval(
                hidden_states=out,
                query=q,
                ip_k=self.call_ip(k_key, cond_mark, q.device, q.dtype),
                ip_v=self.call_ip(v_key, cond_mark, q.device, q.dtype),
                attn_heads=h,
                head_dim=head_dim,
            )
            return self.apply_effective_region_mask(ip_out * weight)

        return forward
