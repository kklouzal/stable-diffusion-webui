import torch
import torch.nn as nn
from collections import OrderedDict

from modules import devices
from sgm.modules.diffusionmodules.openaimodel import Downsample


class PlugableAdapter(nn.Module):
    def __init__(self, control_model) -> None:
        super().__init__()
        self.control_model = control_model
        self.control = None
        self.hint_cond = None
        self.hint_source = None

    def reset(self):
        self.control = None
        self.hint_cond = None
        self.hint_source = None

    def release_request_state(self):
        """Drop the request's hint and features (UnetHook.restore); the model stays cached."""
        self.reset()

    def forward(self, hint=None, x=None, *args, **kwargs):
        """The adapter features of `hint` (a list of tensors, or one tensor for the style adapter), computed
        once per hint object: the hook passes the same hint tensor every step and a different one for the
        hires pass. Callers must not modify the returned tensors in place (hook.py scales them into new
        tensors first)."""
        if self.control is None or hint is not self.hint_source:
            self.hint_source = hint
            self.hint_cond = devices.cond_cast_unet(hint)
            hint_in = self.hint_cond

            if hasattr(self.control_model, 'conv_in') and \
                    (self.control_model.conv_in.in_channels == 64 or self.control_model.conv_in.in_channels == 256):
                hint_in = hint_in[:, 0:1, :, :]

            self.control = self.control_model(hint_in)
        return list(self.control) if isinstance(self.control, list) else self.control

    def aggressive_lowvram(self):
        self.to(devices.get_device_for("controlnet"))
        return

    def fullvram(self):
        self.to(devices.get_device_for("controlnet"))
        return


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize//2
        if in_c != out_c or sk is False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk is False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down is True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down is True:
            x = self.down_opt(x)
        if self.in_conv is not None: # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class Adapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=False, use_conv=True, is_sdxl=True):
        super(Adapter, self).__init__()

        if is_sdxl:
            self.pixel_shuffle = 16
            downsample_avoided = [1]
            downsample_layers = [2]
        else:
            self.pixel_shuffle = 8
            downsample_avoided = []
            downsample_layers = [3, 2, 1]

        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []

        self.unshuffle = nn.PixelUnshuffle(self.pixel_shuffle)

        for i in range(len(channels)):
            for r in range(nums_rb):

                if i in downsample_layers and r == 0:
                    self.body.append(ResnetBlock(
                        channels[i - 1],
                        channels[i],
                        down=True,
                        ksize=ksize,
                        sk=sk,
                        use_conv=use_conv))
                    continue

                if i in downsample_avoided and r == 0:
                    self.body.append(ResnetBlock(
                        channels[i - 1],
                        channels[i],
                        down=False,
                        ksize=ksize,
                        sk=sk,
                        use_conv=use_conv))
                    continue

                self.body.append(ResnetBlock(
                    channels[i],
                    channels[i],
                    down=False,
                    ksize=ksize,
                    sk=sk,
                    use_conv=use_conv
                ))

        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    def forward(self, x):
        self.to(x.device)

        x = self.unshuffle(x)
        hs = []

        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for r in range(self.nums_rb):
                idx = i * self.nums_rb + r
                x = self.body[idx](x)
            hs.append(x)

        self.to('cpu')
        return hs


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class StyleAdapter(nn.Module):

    def __init__(self, width=1024, context_dim=768, num_head=8, n_layes=3, num_token=4):
        super().__init__()

        scale = width ** -0.5
        self.transformer_layes = nn.Sequential(*[ResidualAttentionBlock(width, num_head) for _ in range(n_layes)])
        self.num_token = num_token
        self.style_embedding = nn.Parameter(torch.randn(1, num_token, width) * scale)
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, context_dim))

    def forward(self, x):
        # x shape [N, HW+1, C]
        style_embedding = self.style_embedding + torch.zeros(
            (x.shape[0], self.num_token, self.style_embedding.shape[-1]), device=x.device)
        
        x = torch.cat([x, style_embedding], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_layes(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, -self.num_token:, :])
        x = x @ self.proj

        return x


class ResnetBlock_light(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.block1 = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(in_c, in_c, 3, 1, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)

        return h + x


class extractor(nn.Module):
    def __init__(self, in_c, inter_c, out_c, nums_rb, down=False):
        super().__init__()
        self.in_conv = nn.Conv2d(in_c, inter_c, 1, 1, 0)
        self.body = []
        for _ in range(nums_rb):
            self.body.append(ResnetBlock_light(inter_c))
        self.body = nn.Sequential(*self.body)
        self.out_conv = nn.Conv2d(inter_c, out_c, 1, 1, 0)
        self.down = down
        if self.down is True:
            self.down_opt = Downsample(in_c, use_conv=False)

    def forward(self, x):
        if self.down is True:
            x = self.down_opt(x)
        x = self.in_conv(x)
        x = self.body(x)
        x = self.out_conv(x)

        return x


class Adapter_light(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64):
        super(Adapter_light, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            if i == 0:
                self.body.append(extractor(in_c=cin, inter_c=channels[i]//4, out_c=channels[i], nums_rb=nums_rb, down=False))
            else:
                self.body.append(extractor(in_c=channels[i-1], inter_c=channels[i]//4, out_c=channels[i], nums_rb=nums_rb, down=True))
        self.body = nn.ModuleList(self.body)

    def forward(self, x):
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        for i in range(len(self.channels)):
            x = self.body[i](x)
            features.append(x)

        return features
