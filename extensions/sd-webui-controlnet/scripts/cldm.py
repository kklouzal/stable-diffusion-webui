from typing import List
import torch
import torch.nn as nn

from modules import devices, openclaw_nhwc_groupnorm, shared
from scripts.controlnet_core.controlnet_union import ControlAddEmbedding, ResBlockUnionControlnet

try:
    from sgm.modules.diffusionmodules.openaimodel import conv_nd, linear, zero_module, timestep_embedding, \
        TimestepEmbedSequential, ResBlock, Downsample, SpatialTransformer, exists
    using_sgm = True
except ImportError:
    from ldm.modules.diffusionmodules.openaimodel import conv_nd, linear, zero_module, timestep_embedding, \
        TimestepEmbedSequential, ResBlock, Downsample, SpatialTransformer, exists
    using_sgm = False


def _loaded_value(value, expected_dtype, dtype):
    """`value` bit for bit as the former path left it: copied into a parameter
    of `expected_dtype` by load_state_dict, then Module.to(dtype)."""
    value = value.to(expected_dtype)
    if dtype is not None and expected_dtype.is_floating_point:
        value = value.to(dtype)
    return value


def controlnet_from_state_dict(config, state_dict, dtype=None):
    """Build ControlNet(**config) holding exactly the checkpoint's weights, cast
    as ControlNet(**config) + load_state_dict + .to(dtype) did, without the
    random init of ~1.25B fp32 parameters and the extra full copies. The
    checkpoint must match the architecture guessed from it key for key."""
    with torch.device("meta"):
        model = ControlNet(**config)
    expected = model.state_dict()
    missing = sorted(expected.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - expected.keys())
    if missing or unexpected:
        raise RuntimeError(
            "ControlNet checkpoint does not match the architecture guessed from it: "
            f"{len(missing)} missing keys {missing[:8]}, {len(unexpected)} unexpected keys {unexpected[:8]}")
    model.load_state_dict(
        {key: _loaded_value(value, expected[key].dtype, dtype) for key, value in state_dict.items()},
        strict=True, assign=True)
    unloaded = [name for name, tensor in (*model.named_parameters(), *model.named_buffers()) if tensor.is_meta]
    if unloaded:
        raise RuntimeError(f"ControlNet tensors not provided by the checkpoint: {unloaded[:8]}")
    return model


class PlugableControlModel(nn.Module):
    def __init__(self, config, state_dict=None, dtype=None):
        super().__init__()
        self.config = config
        if state_dict is None:
            self.control_model = ControlNet(**self.config).cpu()
        else:
            self.control_model = controlnet_from_state_dict(self.config, state_dict, dtype)
        self.gpu_component = None
        self.lowvram_hooks = None
        self.is_control_lora = False

    def reset(self):
        pass

    def forward(self, *args, **kwargs):
        return self.control_model(*args, **kwargs)

    def aggressive_lowvram(self):
        """Keep the model on the CPU and move one submodule at a time to the device as it runs.

        Called before every ControlNet forward of a low-VRAM unit. The pre-hooks are registered once per
        model (this model object is cached across requests), not once per call.
        """
        self.to('cpu')
        # Low-VRAM units keep the checkpoint's NCHW weights (restored if a full-VRAM request converted them).
        openclaw_nhwc_groupnorm.apply_controlnet_layout(self, channels_last=False)
        self.gpu_component = None
        if self.lowvram_hooks is not None:
            return

        def send_me_to_gpu(module, _):
            if self.gpu_component == module:
                return

            if self.gpu_component is not None:
                self.gpu_component.to('cpu')

            module.to(devices.get_device_for("controlnet"))
            self.gpu_component = module

        cm = self.control_model
        modules = [cm.time_embed, cm.input_hint_block, cm.label_emb, *cm.input_blocks, *cm.zero_convs, cm.middle_block, cm.middle_block_out]
        # Union ControlNet parts (task_embedding is a bare parameter; union_controlnet_merge moves its row).
        modules += [m for m in (getattr(cm, 'transformer_layes', None), getattr(cm, 'spatial_ch_projs', None), getattr(cm, 'control_add_embedding', None)) if m is not None]
        self.lowvram_hooks = [m.register_forward_pre_hook(send_me_to_gpu) for m in modules]
        return

    def fullvram(self):
        # A low-VRAM request may have left its pre-hooks on this cached model; they would move a
        # submodule back to the CPU before every module call.
        if self.lowvram_hooks is not None:
            for handle in self.lowvram_hooks:
                handle.remove()
            self.lowvram_hooks = None
            self.gpu_component = None
        self.to(devices.get_device_for("controlnet"))
        # NHWC GroupNorm switch, controlnet scope: channels_last weights like the --opt-channelslast SD model's, so the
        # ControlNet's activations stay NHWC; restores the checkpoint layout (bit for bit) once the scope is off.
        # Control-LoRA models patch their layers' weights per forward and keep their layout.
        channels_last = not self.is_control_lora and bool(getattr(shared.cmd_opts, "opt_channelslast", False))
        openclaw_nhwc_groupnorm.apply_controlnet_layout(self, channels_last=channels_last)
        return


class ControlNet(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=True,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_spatial_transformer=True,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=False,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        transformer_depth_middle=None,
        union_controlnet_num_control_type=None,
        device=None,
        global_average_pooling=False,
    ):
        super().__init__()

        self.global_average_pooling = global_average_pooling

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        if transformer_depth_middle is None:
            transformer_depth_middle = transformer_depth[-1]
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            self.num_res_blocks = num_res_blocks

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim, dtype=self.dtype, device=device),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim, dtype=self.dtype, device=device),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1, dtype=self.dtype, device=device)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
                    conv_nd(dims, hint_channels, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 16, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                    nn.SiLU(),
                    conv_nd(dims, 32, 32, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                    nn.SiLU(),
                    conv_nd(dims, 96, 96, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                    nn.SiLU(),
                    zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth[level], context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

        if union_controlnet_num_control_type is not None:
            self.num_control_type = union_controlnet_num_control_type
            num_trans_channel = 320
            num_trans_head = 8
            num_trans_layer = 1
            num_proj_channel = 320
            self.task_embedding = nn.Parameter(torch.empty(
                self.num_control_type, num_trans_channel, dtype=self.dtype, device=device
            ))

            self.transformer_layes = nn.Sequential(*[
                ResBlockUnionControlnet(
                    num_trans_channel, num_trans_head, dtype=self.dtype, device=device
                )
                for _ in range(num_trans_layer)
            ])
            self.spatial_ch_projs = nn.Linear(
                num_trans_channel, num_proj_channel, dtype=self.dtype, device=device
            )

            control_add_embed_dim = 256
            self.control_add_embedding = ControlAddEmbedding(
                control_add_embed_dim, time_embed_dim, self.num_control_type,
                dtype=self.dtype, device=device
            )
        else:
            self.task_embedding = None
            self.control_add_embedding = None

    def union_controlnet_merge(
        self,
        hint: torch.Tensor,
        control_type: List[int],
        emb: torch.Tensor,
        context: torch.Tensor
    ):
        """ Note: control_type is a list of enum values. The length of the list
        is the number of control images."""
        # Equivalent to: https://github.com/xinsir6/ControlNetPlus/tree/main
        inputs = []
        condition_list = []

        for idx in range(min(1, len(control_type))):
            controlnet_cond = self.input_hint_block(hint[idx], emb, context)
            feat_seq = torch.mean(controlnet_cond, dim=(2, 3))
            if idx < len(control_type):
                # No-op except in low-VRAM mode, where the bare parameter stays on the CPU.
                feat_seq += self.task_embedding[control_type[idx]].to(device=feat_seq.device)

            inputs.append(feat_seq.unsqueeze(1))
            condition_list.append(controlnet_cond)

        x = torch.cat(inputs, dim=1)
        x = self.transformer_layes(x)
        controlnet_cond_fuser = None
        for idx in range(len(control_type)):
            alpha = self.spatial_ch_projs(x[:, idx])
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
            o = condition_list[idx] + alpha
            if controlnet_cond_fuser is None:
                controlnet_cond_fuser = o
            else:
                controlnet_cond_fuser += o
        return controlnet_cond_fuser

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def compute_guided_hint(self, hint, control_type: List[int] = None):
        """Hint embedding added after the first input block.

        A function of the hint and the union control types only: the hint block is convolutions and SiLU,
        which TimestepEmbedSequential calls without emb/context, and the union merge mixes per-type task
        embeddings. Callers may compute it once per hint and pass it to forward as ``guided_hint``.
        """
        hint = hint.to(devices.dtype_unet)
        if self.control_add_embedding is not None:
            assert control_type is not None
            if len(control_type) > 0:
                if len(hint.shape) < 5:
                    hint = hint.unsqueeze(dim=0)
                return self.union_controlnet_merge(hint, control_type, None, None)
        return self.input_hint_block(hint, None, None)

    def forward(self, x, hint, timesteps, context, y=None, control_type: List[int] = None, guided_hint=None, **kwargs):
        original_type = x.dtype
        # Compute in the UNet dtype the weights are converted to when the model is built
        # (controlnet_model_guess); self.dtype is only the construction dtype and stays float32 under
        # bfloat16, which made every input an fp32 copy that autocast cast back per layer.
        dtype = devices.dtype_unet

        x = x.to(dtype)
        # Timesteps stay float32 like the UNet's (sd_hijack_unet.apply_model): a half-precision cast would quantize them.
        timesteps = timesteps.to(torch.float32)
        context = context.to(dtype)

        if y is not None:
            y = y.to(dtype)

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(dtype)
        emb = self.time_embed(t_emb)

        if self.control_add_embedding is not None:
            assert control_type is not None
            emb += self.control_add_embedding(control_type, emb.dtype, emb.device)

        if guided_hint is None:
            guided_hint = self.compute_guided_hint(hint, control_type)

        outs = []

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        outs = [o.to(original_type) for o in outs]

        return outs
