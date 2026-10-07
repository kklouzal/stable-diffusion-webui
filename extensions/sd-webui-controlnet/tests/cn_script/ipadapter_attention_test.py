import importlib
import types
import unittest

import torch
import torch.nn.functional as F

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from scripts.ipadapter import plugable_ipadapter  # noqa: E402
from scripts.ipadapter.ipadapter_model import ImageEmbed, IPAdapterModel, To_KV  # noqa: E402
from scripts.ipadapter.pulid_attn import PULID_SETTING_FIDELITY, PULID_SETTING_STYLE  # noqa: E402

HEADS, HEAD_DIM, CONTEXT_DIM, TOKENS = 2, 8, 12, 4
INNER = HEADS * HEAD_DIM
LATENT_H, LATENT_W = 4, 6
SEQ = LATENT_H * LATENT_W


class Attn(torch.nn.Module):
    """The parameters of an SDXL attn2 (sgm CrossAttention): bias-free q/k/v, to_out = Linear + Dropout."""

    def __init__(self):
        super().__init__()
        self.heads = HEADS
        self.to_q = torch.nn.Linear(INNER, INNER, bias=False)
        self.to_k = torch.nn.Linear(CONTEXT_DIM, INNER, bias=False)
        self.to_v = torch.nn.Linear(CONTEXT_DIM, INNER, bias=False)
        self.to_out = torch.nn.Sequential(torch.nn.Linear(INNER, INNER), torch.nn.Dropout(0.0))


def split_heads(t):
    return t.view(t.shape[0], -1, HEADS, HEAD_DIM).transpose(1, 2)


def merge_heads(t):
    return t.transpose(1, 2).reshape(t.shape[0], -1, INNER)


def oracle(attn, x, context, w_k, w_v, row_embeds, weight, pulid=None, region=None):
    """One row at a time: IPAttnProcessor2_0 of tencent-ailab/IP-Adapter, or IDAttnProcessor2_0 of
    ToTheBeginning/PuLID (whose orthogonal modes project against the main attention output)."""
    rows = []
    for i, emb in enumerate(row_embeds):
        q = split_heads(attn.to_q(x[i:i + 1]))
        hidden = merge_heads(F.scaled_dot_product_attention(
            q, split_heads(attn.to_k(context[i:i + 1])), split_heads(attn.to_v(context[i:i + 1]))))
        if pulid is not None and pulid.num_zero:
            emb = torch.cat([emb, torch.zeros(1, pulid.num_zero, emb.shape[-1])], dim=1)
        key, value = split_heads(F.linear(emb, w_k)), split_heads(F.linear(emb, w_v))
        ip = merge_heads(F.scaled_dot_product_attention(q, key, value))
        projection = (hidden * ip).sum(-2, keepdim=True) / (hidden * hidden).sum(-2, keepdim=True) * hidden
        if pulid is not None and pulid.ortho_v2:
            attn_mean = (q @ key.transpose(-2, -1)).softmax(dim=-1).mean(dim=1)[:, :, :5].sum(dim=-1, keepdim=True)
            ip = ip + (attn_mean - 1) * projection
        elif pulid is not None and pulid.ortho:
            ip = ip - projection
        ip = ip * weight
        if region is not None:
            ip = ip * F.interpolate(region, size=(LATENT_H, LATENT_W), mode="bilinear").view(1, -1, 1)
        rows.append(attn.to_out(hidden + ip))
    return torch.cat(rows)


class TestIPAdapterAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.attn = Attn()
        self.w_k = torch.randn(INNER, CONTEXT_DIM) / CONTEXT_DIM ** 0.5
        self.w_v = torch.randn(INNER, CONTEXT_DIM) / CONTEXT_DIM ** 0.5
        self.cond = torch.randn(1, TOKENS, CONTEXT_DIM)
        self.uncond = torch.randn(1, TOKENS, CONTEXT_DIM)
        self.to_kv = To_KV({"1.to_k_ip.weight": self.w_k, "1.to_v_ip.weight": self.w_v})

    def tearDown(self):
        plugable_ipadapter.clear_all_ip_adapter()

    def adapter(self, weight=0.7, pulid=None, region=None, transformer_index=0):
        ip = plugable_ipadapter.PlugableIPAdapter(
            types.SimpleNamespace(ip_layers=self.to_kv, is_pulid=pulid is not None))
        ip.reset()
        ip.image_emb = ImageEmbed(self.cond, self.uncond)
        ip.weight, ip.p_start, ip.p_end = weight, 0.0, 1.0
        ip.pulid_attn_setting = pulid
        ip.effective_region_mask = region
        ip.latent_width, ip.latent_height = LATENT_W, LATENT_H
        plugable_ipadapter.hack_blk(self.attn, ip.patch_forward(0, transformer_index), Attn)
        return ip

    def call(self, rows, **oracle_kwargs):
        """One UNet call whose rows are cond (True) or uncond (False), against the per-row oracle."""
        x = torch.randn(len(rows), SEQ, INNER)
        context = torch.randn(len(rows), 7, CONTEXT_DIM)
        cond_mark = torch.tensor([1.0 if r else 0.0 for r in rows]).view(-1, 1, 1, 1)
        plugable_ipadapter.current_model = types.SimpleNamespace(cond_mark=cond_mark, current_sampling_percent=0.5)
        with torch.no_grad():
            actual = self.attn(x, context)
            expected = oracle(self.attn, x, context, self.w_k, self.w_v,
                              [self.cond if r else self.uncond for r in rows], **oracle_kwargs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_image_kv_rows_follow_every_call(self):
        """A1111 runs cond and uncond rows as separate calls (prompts of different token lengths, batch_cond_uncond
        off) and drops the uncond rows on skipped-uncond steps; the k/v cache must not freeze the first call's rows."""
        for region in (None, torch.rand(1, 1, 2 * LATENT_H, 2 * LATENT_W)):
            with self.subTest(region=region is not None):
                self.adapter(region=region)
                for rows in ([True, False], [True], [False], [True, True, False, False], [False, True]):
                    self.call(rows, weight=0.7, region=region)
                plugable_ipadapter.clear_all_ip_adapter()

    def test_pulid_projects_against_attention_output(self):
        for setting in (PULID_SETTING_FIDELITY, PULID_SETTING_STYLE):
            with self.subTest(setting=setting):
                self.adapter(weight=0.8, pulid=setting)
                for rows in ([True, False], [False]):
                    self.call(rows, weight=0.8, pulid=setting)
                plugable_ipadapter.clear_all_ip_adapter()

    def test_image_embeds_projected_once_per_request(self):
        counts = {}
        for name, module in self.to_kv.to_kvs.items():
            module.register_forward_hook(lambda m, i, o, name=name: counts.__setitem__(name, counts.get(name, 0) + 1))
        self.adapter()
        for rows in ([True, False], [True], [False], [True, False]):
            self.call(rows, weight=0.7)
        self.assertEqual(counts, {"1_to_k_ip": 1, "1_to_v_ip": 1})

    def test_inactive_layer_leaves_attention_unchanged(self):
        x, context = torch.randn(2, SEQ, INNER), torch.randn(2, 7, CONTEXT_DIM)
        plugable_ipadapter.current_model = types.SimpleNamespace(
            cond_mark=torch.tensor([1.0, 0.0]).view(-1, 1, 1, 1), current_sampling_percent=0.5)
        with torch.no_grad():
            plain = merge_heads(F.scaled_dot_product_attention(
                split_heads(self.attn.to_q(x)), split_heads(self.attn.to_k(context)), split_heads(self.attn.to_v(context))))
            expected = self.attn.to_out(plain)
            self.adapter(weight={3: 1.0}, transformer_index=0)
            self.assertTrue(torch.equal(self.attn(x, context), expected))


class TestIPAdapterPlusUncond(unittest.TestCase):
    def test_uncond_is_the_models_projection_of_the_zero_image(self):
        """IPAdapterPlus/IPAdapterPlusXL.get_image_embeds (tencent-ailab/IP-Adapter) project the ViT-H penultimate
        hidden states of a zero image with the loaded model's own image_proj_model, for SDXL as for SD1.5."""
        from annotator.clipvision import clip_vision_h_uc

        torch.manual_seed(0)
        for sdxl_plus in (False, True):
            with self.subTest(sdxl_plus=sdxl_plus):
                model = IPAdapterModel.__new__(IPAdapterModel)
                torch.nn.Module.__init__(model)
                model.device, model.is_plus, model.sdxl_plus = "cpu", True, sdxl_plus
                model.image_proj_model = torch.nn.Sequential(torch.nn.Linear(1280, 32), torch.nn.LayerNorm(32))
                hidden = torch.randn(1, 257, 1280)
                embed = model._get_image_embeds({"hidden_states": [hidden, hidden, hidden]})
                with torch.inference_mode():
                    expected = model.image_proj_model(clip_vision_h_uc.to(device="cpu", dtype=torch.float32))
                torch.testing.assert_close(embed.uncond_emb, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
