#!/usr/bin/env python3
"""Patch and verify the GB10 changes to the mounted MultiDiffusion / Tiled VAE extension.

Each change is an exact ORIGINAL -> PATCHED text block (gb10/patchlib.py: a target is either fully original and gets
patched, or fully patched and is left alone; anything else, CRLF included, aborts the deploy; nothing is written until
every target verifies; --check writes nothing). ORIGINAL is upstream origin/main 22798f6, so a fresh install patches
cleanly. The blocks:
- MD terminal tile origins (tile_utils/utils.py): upstream spreads origins as int(col * (w - tile_w) / (cols - 1));
  the float floor can put the last one at w - tile_w - 1 and leave the last latent column or row uncovered
  (weight 0). The replacement keeps upstream's count and origins and pins the last origin to w - tile_w. Over every
  tile 5..256, overlap 0..tile-4 and extent tile+1..1024 it equals upstream wherever upstream covers the extent and
  otherwise moves only the last origin, by one. A tile that spans the extent is one origin (upstream divides by zero
  when init_grid_bbox's clamped overlap equals it). The deploy10 release stepped by tile - overlap instead, which
  shrank the seam overlaps against upstream's even spacing for most extents; PREVIOUS upgrades it.
- TV-FB: the former local commit "Modernize Tiled VAE attention fallbacks" (5022f68): xformers is optional and falls
  back to SDP, and the sage2/sage3 method names map to SDP. Its torch.nn.attention import is no longer used since
  TV-ATTN; it is kept, like the sage names, so the deployed bytes stay unchanged.
- TV-*, MD-W1, MD-CN: performance changes. Exactness of each is argued next to the code it patches and tested on CPU
  against the unpatched extension (test/test_gb10_multidiffusion_performance_patcher.py). MD-CN builds each ControlNet
  control tile once per request and skips reassigning an unchanged one; ControlNet still re-derives (and hashes) the
  hint whenever the tile changes, i.e. per tile batch per step with more than one batch.
- TV-GN, TV-NORM, TV-ENC8 (Tiled VAE results): non-fast group-norm statistics are pooled exactly over each tile's
  valid region (upstream took whole padded tiles and dropped the between-tile variance: real SDXL VAE 640x768 decode,
  tile 48, PSNR vs untiled 47.1 -> 59.0 dB); the normalize keeps float32 statistics and affine and rounds once (bf16
  no worse than native GroupNorm); encoder tile sizes floor to a multiple of 8 (off-grid tiles misplaced the latent
  or failed). Fast mode keeps its documented estimated statistics. TV-NANEND checks NaNs once per finished tile.
- MD-NI-COND: noise inversion encodes this batch's prompts with extra networks parsed out, as SdConditioning with the
  canvas size (SDXL embeds it), instead of the first batch's raw prompts as a plain list.
- MD-NI-CACHE: the inverted-noise cache lives for one request and is reused only for exact matches (see the block),
  for Tiled Diffusion (scripts/tilediffusion.py) and DemoFusion (scripts/tileglobal.py) alike: both build the cache entry.
- MD-REGION-COND: region prompt control on SDXL/SD3 fails in Script.process(), before any work, instead of with a
  TypeError once sampling has started.
A checkout holding only the former 0001 commit (upstream + TV-FB, nothing else) is not accepted: reset it to upstream.
"""
from __future__ import annotations

from patchlib import Block, apply_blocks, parse_cli

LABEL = "MultiDiffusion"
TILEVAE = "scripts/tilevae.py"
TILEVAE_DEPLOY10_BLOCKS = 8  # TV-GC .. TV-APX return; the TV-NORM, TV-GN, TV-ENC8 and TV-NANEND blocks came after

BLOCKS: dict[str, list[Block]] = {
    "tile_utils/utils.py": [
        Block(
            "MD terminal tile origins",
            '''def split_bboxes(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    cols = math.ceil((w - overlap) / (tile_w - overlap))
    rows = math.ceil((h - overlap) / (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=devices.device, dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)

            bbox = BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight
''',
            '''def _gb10_tile_origins(extent:int, tile:int, overlap:int) -> List[int]:
    # gb10: upstream's origins with the last one pinned to extent - tile (see gb10/patch-multidiffusion-performance.py).
    if tile >= extent:
        return [0]
    if not 0 <= overlap < tile:
        raise ValueError(f"tile overlap must be in [0, tile), got overlap={overlap}, tile={tile}")
    count = math.ceil((extent - overlap) / (tile - overlap))
    step = (extent - tile) / (count - 1)
    return [int(i * step) for i in range(count - 1)] + [extent - tile]


def split_bboxes(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    x_origins = _gb10_tile_origins(w, tile_w, overlap)
    y_origins = _gb10_tile_origins(h, tile_h, overlap)

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=devices.device, dtype=torch.float32)
    for y in y_origins:
        for x in x_origins:
            bbox = BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight
''',
            sentinel="def _gb10_tile_origins",
        ),
        Block("MD-NI-CACHE key", r"""NoiseInverseCache = namedtuple('NoiseInversionCache', ['model_hash', 'x0', 'xt', 'noise_inversion_steps', 'retouch', 'prompts'])
""", r"""NoiseInverseCache = namedtuple('NoiseInversionCache', ['model_hash', 'x0', 'xt', 'noise_inversion_steps', 'retouch', 'prompts', 'extra_network_data'])  # gb10: MD-NI-CACHE
"""),
    ],
    TILEVAE: [
        Block(
            "TV-GC perfcount",
            r"""        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(devices.device)
        devices.torch_gc()
        gc.collect()

        ret = fn(*args, **kwargs)

        devices.torch_gc()
        gc.collect()
        if torch.cuda.is_available():
""",
            r"""        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(devices.device)

        # gb10 (TV-GC): no torch_gc()/gc.collect() around the call. On unified memory empty_cache() frees nothing
        # useful and only makes the allocator regrow multi-GB blocks; the allocator itself frees cached blocks and
        # retries before an allocation fails.
        ret = fn(*args, **kwargs)

        if torch.cuda.is_available():
""",
        ),
        Block(
            "TV-NAN estimation check",
            r"""        # estimate until the last group norm
        for i in range(last_id + 1):
            task = task_queue[i]
            if task[0] == 'pre_norm':
                group_norm_func = GroupNormParam.from_tile(tile, task[1])
                task_queue[i] = ('apply_norm', group_norm_func)
                if i == last_id:
                    return True
                tile = group_norm_func(tile)
            elif task[0] == 'store_res':
                task_id = i + 1
                while task_id < last_id and task_queue[task_id][0] != 'add_res':
                    task_id += 1
                if task_id >= last_id:
                    continue
                task_queue[task_id][1] = task[1](tile)
            elif task[0] == 'add_res':
                tile += task[1].to(device)
                task[1] = None
            elif color_fix and task[0] == 'downsample':
                for j in range(i, last_id + 1):
                    if task_queue[j][0] == 'store_res':
                        task_queue[j] = ('store_res_cpu', task_queue[j][1])
                return True
            else:
                tile = task[1](tile)
            try:
                devices.test_for_nans(tile, "vae")
            except:
                print(f'Nan detected in fast mode estimation. Fast mode disabled.')
                return False

        raise IndexError('Should not reach here')
""",
            r"""        # gb10 (TV-NAN): one NaN check of the final estimation tile instead of a GPU sync after every task.
        # test_for_nans reads element [0,0,0,0]. A NaN there survives every later estimation op at [0,0,0,0]:
        # - group norm from the tile's own statistics makes the whole group NaN;
        # - SiLU, residual adds and nearest upsampling are elementwise or copy;
        # - padded 3x3, strided downsample and 1x1 convs all read input [.,0,0] for output [.,0,0];
        # - attention's query 0 mixes in every channel at position 0.
        # The final tile was also checked before, so the fast-mode decision is unchanged.
        def no_nans(tile):
            try:
                devices.test_for_nans(tile, "vae")
            except:
                print(f'Nan detected in fast mode estimation. Fast mode disabled.')
                return False
            return True

        # estimate until the last group norm
        for i in range(last_id + 1):
            task = task_queue[i]
            if task[0] == 'pre_norm':
                group_norm_func = GroupNormParam.from_tile(tile, task[1])
                task_queue[i] = ('apply_norm', group_norm_func)
                if i == last_id:
                    return no_nans(tile)
                tile = group_norm_func(tile)
            elif task[0] == 'store_res':
                task_id = i + 1
                while task_id < last_id and task_queue[task_id][0] != 'add_res':
                    task_id += 1
                if task_id >= last_id:
                    continue
                task_queue[task_id][1] = task[1](tile)
            elif task[0] == 'add_res':
                tile += task[1].to(device)
                task[1] = None
            elif color_fix and task[0] == 'downsample':
                for j in range(i, last_id + 1):
                    if task_queue[j][0] == 'store_res':
                        task_queue[j] = ('store_res_cpu', task_queue[j][1])
                return no_nans(tile)
            else:
                tile = task[1](tile)

        raise IndexError('Should not reach here')
""",
        ),
        Block(
            "TV-CPU input tiles",
            r"""        tiles = []
        for input_bbox in in_bboxes:
            tile = z[:, :, input_bbox[2]:input_bbox[3], input_bbox[0]:input_bbox[1]].cpu()
            tiles.append(tile)
""",
            r"""        # gb10 (TV-CPU): input tiles, residuals and parked tiles stay on the device. On unified memory the .cpu()
        # round trips saved no memory and cost synchronous copies; copies never change values. Input tiles are views
        # of z, which is safe because their first task, conv_in, reads them out of place.
        tiles = []
        for input_bbox in in_bboxes:
            tile = z[:, :, input_bbox[2]:input_bbox[3], input_bbox[0]:input_bbox[1]]
            tiles.append(tile)
""",
        ),
        Block(
            "TV-APX lazy approximation",
            r"""        # Dummy result
        result = None
        result_approx = None
        try:
            with devices.autocast():
                result_approx = torch.cat([F.interpolate(cheap_approximation(x).unsqueeze(0), scale_factor=opt_f, mode='nearest-exact') for x in z], dim=0).cpu()
        except: pass
        # Free memory of input latent tensor
        del z
""",
            r"""        # Dummy result
        result = None
        # gb10 (TV-APX): z is kept for the cheap approximation. It is only returned when an interrupt stops the run
        # before any tile finished, so it is now only built then, at the end.
""",
        ),
        Block(
            "TV-CPU residuals",
            r"""                        res = task[1](tile)
                        if not self.fast_mode or task[0] == 'store_res_cpu':
                            res = res.cpu()
""",
            r"""                        # gb10 (TV-CPU): an identity residual now aliases the tile. That is safe: the next op on the
                        # tile is always a group norm, which writes a new tensor.
                        res = task[1](tile)
""",
        ),
        Block(
            "TV-RES result dtype",
            r"""                    if result is None:      # NOTE: dim C varies from different cases, can only be inited dynamically
                        result = torch.zeros((N, tile.shape[1], height * 8 if is_decoder else height // 8, width * 8 if is_decoder else width // 8), device=device, requires_grad=False)
""",
            r"""                    if result is None:      # NOTE: dim C varies from different cases, can only be inited dynamically
                        # gb10 (TV-RES): allocate in the VAE dtype. fp32 holds fp16/bf16/fp32 tiles exactly, so
                        # storing them in fp32 and converting once at the end rounded exactly like storing them directly.
                        result_dtype = dtype if tile.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
                        result = torch.zeros((N, tile.shape[1], height * 8 if is_decoder else height // 8, width * 8 if is_decoder else width // 8), device=device, dtype=result_dtype, requires_grad=False)
""",
        ),
        Block(
            "TV-CPU parked tiles",
            r"""                else:
                    tiles[i] = tile.cpu()
                    del tile
""",
            r"""                else:
                    tiles[i] = tile  # gb10 (TV-CPU): parked on the device
                    del tile
""",
        ),
        Block(
            "TV-APX return",
            r"""        # Done!
        pbar.close()
        return result.to(dtype) if result is not None else result_approx.to(device, dtype=dtype)
""",
            r"""        # Done!
        pbar.close()
        if result is None:
            with devices.autocast():
                result = torch.cat([F.interpolate(cheap_approximation(x).unsqueeze(0), scale_factor=opt_f, mode='nearest-exact') for x in z], dim=0)
            return result.to(device, dtype=dtype)
        return result.to(dtype)
""",
        ),
        Block(
            'TV-NORM fused fp32 normalize',
            r"""    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:])

    out = F.batch_norm(input_reshaped, mean.to(input), var.to(input), weight=None, bias=None, training=False, momentum=0, eps=eps)
    out = out.view(b, c, *input.size()[2:])

    # post affine transform
    if weight is not None:
        out *= weight.view(1, -1, 1, 1)
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    return out
""",
            r"""    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    # gb10 (TV-NORM): one batch_norm over the b*c channels of a [1, b*c, H, W] view, with the per-(sample, group)
    # statistics expanded to their channels and the affine folded in. Statistics and affine stay float32 (batch_norm
    # computes in float32 for half/bfloat16 input) and the output is rounded once, like F.group_norm. The former
    # path rounded the statistics to the tile dtype and rounded again after the normalize, the scale and the shift.
    def per_channel(t):
        return t.float().view(b, num_groups, 1).expand(b, num_groups, channel_in_group).reshape(b * c)
    out = F.batch_norm(input.contiguous().view(1, b * c, *input.size()[2:]), per_channel(mean), per_channel(var),
                       weight=None if weight is None else weight.float().repeat(b),
                       bias=None if bias is None else bias.float().repeat(b), training=False, momentum=0, eps=eps)
    return out.view(b, c, *input.size()[2:])
""",
        ),
        Block(
            'TV-GN crop margins at any resolution',
            r"""    padded_bbox = [i * 8 if is_decoder else i//8 for i in input_bbox]
    margin = [target_bbox[i] - padded_bbox[i] for i in range(4)]
    return x[:, :, margin[2]:x.size(2)+margin[3], margin[0]:x.size(3)+margin[1]]
""",
            r"""    padded_bbox = [i * 8 if is_decoder else i//8 for i in input_bbox]
    # gb10 (TV-GN): x may also be an intermediate activation of the tile (group norm statistics): scale the output-space
    # margins by x's size relative to the final tile. Decoder margins are multiples of 8 and encoder tiles start on the
    # 8-pixel grid (TV-ENC8), so the scaling is exact; for the final tile it is the identity.
    h, w = padded_bbox[3] - padded_bbox[2], padded_bbox[1] - padded_bbox[0]
    margin = [(target_bbox[i] - padded_bbox[i]) * (x.size(3) if i < 2 else x.size(2)) // (w if i < 2 else h) for i in range(4)]
    return x[:, :, margin[2]:x.size(2)+margin[3], margin[0]:x.size(3)+margin[1]]
""",
        ),
        Block(
            'TV-GN pooled statistics',
            r"""        var = torch.vstack(self.var_list)
        mean = torch.vstack(self.mean_list)
        max_value = max(self.pixel_list)
        pixels = torch.tensor(self.pixel_list, dtype=torch.float32, device=devices.device) / max_value
        sum_pixels = torch.sum(pixels)
        pixels = pixels.unsqueeze(1) / sum_pixels
        var = torch.sum(var * pixels, dim=0)
        mean = torch.sum(mean * pixels, dim=0)
""",
            r"""        # gb10 (TV-GN): exact pooling of the disjoint valid regions (law of total variance) in float32: the within-tile
        # variances plus the spread of the tile means. Upstream averaged the variances only, over whole padded tiles.
        var = torch.vstack(self.var_list).float()
        mean = torch.vstack(self.mean_list).float()
        pixels = torch.tensor(self.pixel_list, dtype=torch.float32, device=mean.device)
        pixels = (pixels / pixels.sum()).unsqueeze(1)
        mean_all = torch.sum(mean * pixels, dim=0)
        var = torch.sum((var + (mean - mean_all) ** 2) * pixels, dim=0)
        mean = mean_all
""",
        ),
        Block(
            'TV-ENC8 encoder tile grid',
            r"""        self.tile_size = tile_size
""",
            r"""        # gb10 (TV-ENC8): encoder tiles stay on the 8-pixel latent grid (get_best_tile_size keeps multiples of 8 only if
        # tile_size is one; the UI slider steps by 16, API arguments need not).
        self.tile_size = tile_size if is_decoder else max(8, int(tile_size) // 8 * 8)
""",
        ),
        Block(
            'TV-GN valid-region statistics',
            r"""                    if task[0] == 'pre_norm':
                        group_norm_param.add_tile(tile, task[1])
""",
            r"""                    if task[0] == 'pre_norm':
                        # gb10 (TV-GN): statistics of the tile's own output region only; its padding overlaps the
                        # neighbouring tiles (counted 2-4x) and holds the tile-local zero padding.
                        group_norm_param.add_tile(crop_valid_region(tile, in_bboxes[i], out_bboxes[i], is_decoder), task[1])
""",
        ),
        Block(
            'TV-NANEND finished-tile NaN check',
            r"""                # check for NaNs in the tile.
                # If there are NaNs, we abort the process to save user's time
                devices.test_for_nans(tile, "vae")
""",
            r"""                # gb10 (TV-NANEND): check the finished tile only, not after every group-norm pass (a host sync per tile per
                # pass). A NaN anywhere in a tile survives every later op in place (TV-NAN's argument; strided convs read
                # every input position) and a NaN in a valid region poisons the pooled statistics of every tile, so a
                # NaN that the per-pass check caught still reaches the finished tile's [0,0,0,0] check, only later.
                if len(task_queue) == 0:
                    devices.test_for_nans(tile, "vae")
""",
        ),
    ],
    "tile_utils/attn.py": [
        Block(
            "TV-ATTN imports",
            r"""import torch

from modules import shared, sd_hijack
from einops import rearrange
from modules.sd_hijack_optimizations import get_available_vram, get_xformers_flash_attention_op, sub_quad_attention
""",
            r"""import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from modules import shared, sd_hijack
from einops import rearrange
from modules.devices import without_autocast
from modules.sd_hijack_optimizations import get_available_vram, get_xformers_flash_attention_op, run_scaled_dot_product_attention, sub_quad_attention  # gb10: TV-ATTN
""",
        ),
        Block(
            "TV-FB optional xformers",
            r"""    import xformers.ops
except ImportError:
    pass
""",
            r"""    import xformers.ops
except ImportError:
    xformers = None
""",
        ),
        Block(
            "TV-FB method names",
            r"""    # ['none', 'sdp-no-mem', 'sdp', 'xformers', ''sub-quadratic', 'v1', 'invokeai', 'doggettx']
    if method not in ['none', 'sdp-no-mem', 'sdp', 'xformers', 'sub-quadratic', 'v1', 'invokeai', 'doggettx']:
""",
            r"""    # ['none', 'sdp-no-mem', 'sdp', 'xformers', 'sage2', 'sage3', 'sub-quadratic', 'v1', 'invokeai', 'doggettx']
    if method not in ['none', 'sdp-no-mem', 'sdp', 'xformers', 'sage2', 'sage3', 'sub-quadratic', 'v1', 'invokeai', 'doggettx']:
""",
        ),
        Block(
            "TV-FB method fallbacks",
            r"""    elif method == 'xformers':
        return xformers_attnblock_forward
    elif method == 'sdp-no-mem':
        return sdp_no_mem_attnblock_forward
    elif method == 'sdp':
        return sdp_attnblock_forward
""",
            r"""    elif method == 'xformers':
        if xformers is None:
            print("[Tiled VAE] Warning: xformers attention requested but xformers is unavailable; falling back to SDP.")
            return sdp_attnblock_forward
        return xformers_attnblock_forward
    elif method == 'sdp-no-mem':
        return sdp_no_mem_attnblock_forward
    elif method in {'sdp', 'sage2', 'sage3'}:
        # SageAttention A1111 backends do not expose a Tiled VAE AttnBlock path; SDP is the safest fast decode fallback.
        return sdp_attnblock_forward
""",
        ),
        Block(
            "TV-ATTN 4-D SDPA",
            r"""def sdp_no_mem_attnblock_forward(self, x):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        return sdp_attnblock_forward(self, x)

def sdp_attnblock_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    out = out.to(dtype)
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    out = self.proj_out(out)
    return out
""",
            r"""def sdp_no_mem_attnblock_forward(self, x):
    return sdp_attnblock_forward(self, x, sdpa_backend_override="flash,math")

def sdp_attnblock_forward(self, h_, sdpa_backend_override=None):
    # gb10 (TV-ATTN): one head of 4-D [b, 1, hw, c] q/k/v. PyTorch's fused SDPA kernels need 4-D inputs, so the
    # old 3-D [b, hw, c] call always fell back to the math kernel and materialized the hw x hw scores (about 12 GB
    # bf16 for one 278x278 decoder tile). webui's helper also applies its SDPA backend policy. Same attention,
    # different valid kernel: numerically equivalent. Upcasting (upcast_attn) also turns autocast off for the kernel:
    # CUDA autocast runs scaled_dot_product_attention in its lower-precision dtype and would cast the float32 q/k/v
    # straight back (same fix as modules/sd_hijack_optimizations.py); without upcast_attn it is a no-op.
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = (t.reshape(b, 1, c, h * w).transpose(-1, -2) for t in (q, k, v))
    dtype = q.dtype
    upcast = shared.opts.upcast_attn
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    with without_autocast(disable=not upcast):
        out = run_scaled_dot_product_attention(q, k, v, is_causal=False, sdpa_backend_override=sdpa_backend_override)
    out = out.to(dtype)
    out = out.transpose(-1, -2).reshape(b, c, h, w)
    out = self.proj_out(out)
    return out
""",
        ),
    ],
    "tile_methods/abstractdiffusion.py": [
        Block(
            "MD-NI-COND/CACHE noise inversion prompts and cache",
            r"""        prompts = p.all_prompts[:p.batch_size]
        
        latent = None
        # try to use cached latent to save huge amount of time.
        cached_latent: NoiseInverseCache = self.noise_inverse_get_cache()
        if cached_latent is not None and \
            cached_latent.model_hash == p.sd_model.sd_model_hash and \
            cached_latent.noise_inversion_steps == self.noise_inverse_steps and \
            len(cached_latent.prompts) == len(prompts) and \
            all([cached_latent.prompts[i] == prompts[i] for i in range(len(prompts))]) and \
            abs(cached_latent.retouch - self.noise_inverse_retouch) < 0.01 and \
            cached_latent.x0.shape == p.init_latent.shape and \
            torch.abs(cached_latent.x0.to(p.init_latent.device) - p.init_latent).sum() < 100: # the 100 is an arbitrary threshold copy-pasted from the img2img alt code
                # use cached noise
                print('[Tiled Diffusion] Your checkpoint, image, prompts, inverse steps, and retouch params are all unchanged.')
                print('[Tiled Diffusion] Noise Inversion will use the cached noise from the previous run. To clear the cache, click the Free GPU button.')
                latent = cached_latent.xt.to(noise.device)
        if latent is None:
            # run noise inversion
            shared.state.job_count += 1
            latent = self.find_noise_for_image_sigma_adjustment(sampler.model_wrap, self.noise_inverse_steps, prompts)
            shared.state.nextjob()
            self.noise_inverse_set_cache(p.init_latent.clone().cpu(), latent.clone().cpu(), prompts)
            # The cache is only 1 latent image and is very small (16 MB for 8192 * 8192 image), so we don't need to worry about memory leakage.
""",
            r"""        # gb10 (MD-NI-COND): this batch's prompts with the extra networks processing parsed out (p.prompts), not the
        # first batch's raw prompts (<lora:...> tags were encoded as text and every batch inverted the first one's).
        prompts = p.prompts

        latent = None
        # gb10 (MD-NI-CACHE): the inversion depends on everything that shapes the UNet output (checkpoint, LoRA/TI/
        # hypernetwork weights, ControlNet and the other UNet hooks, tiling, noise schedule, attention backend, dtype),
        # so no complete key exists across requests: the cache lives for one request (Script.process clears it). It is
        # reused between that request's batches only for the exact same init latent, prompts, extra networks, steps and
        # retouch, and never with ControlNet (batch inputs can change its hints between batches). Upstream compared the
        # init latents by an absolute sum |dx0| < 100 and retouch within 0.01, and accepted different inputs.
        cached_latent: NoiseInverseCache = self.noise_inverse_get_cache()
        if cached_latent is not None and not self.enable_controlnet and \
            cached_latent.model_hash == p.sd_model.sd_model_hash and \
            cached_latent.noise_inversion_steps == self.noise_inverse_steps and \
            cached_latent.prompts == prompts and \
            cached_latent.extra_network_data == p.extra_network_data and \
            cached_latent.retouch == self.noise_inverse_retouch and \
            cached_latent.x0.shape == p.init_latent.shape and \
            torch.equal(cached_latent.x0.to(p.init_latent.device), p.init_latent):
                print('[Tiled Diffusion] Noise Inversion reuses the inverted noise of the previous batch (same image, prompts and settings).')
                latent = cached_latent.xt.to(noise.device)
        if latent is None:
            # run noise inversion
            shared.state.job_count += 1
            latent = self.find_noise_for_image_sigma_adjustment(sampler.model_wrap, self.noise_inverse_steps, prompts)
            shared.state.nextjob()
            # An interrupted inversion returns its partial latent: never cache it.
            if not shared.state.interrupted:
                self.noise_inverse_set_cache(p.init_latent.clone().cpu(), latent.clone().cpu(), prompts)
""",
        ),
        Block("MD-NI-COND conditioning size", r"""        cond = self.p.sd_model.get_learned_conditioning(prompts)
""", r"""        # gb10 (MD-NI-COND): SdConditioning carries the canvas size, which SDXL embeds (a plain list got 1024x1024).
        cond = self.p.sd_model.get_learned_conditioning(prompt_parser.SdConditioning(prompts, width=self.p.width, height=self.p.height))
"""),
        Block("MD-CN tile memo reset", r"""    @controlnet
    def prepare_controlnet_tensors(self, refresh:bool=False):
        ''' Crop the control tensor into tiles and cache them '''

        if not refresh:
            if self.control_tensor_batch is not None or self.control_params is not None: return

        if not self.enable_controlnet or self.controlnet_script is None: return
""", r"""    @controlnet
    def prepare_controlnet_tensors(self, refresh:bool=False):
        ''' Crop the control tensor into tiles and cache them '''

        if not refresh:
            if self.control_tensor_batch is not None or self.control_params is not None: return

        self.control_tile_device = {}  # gb10 (MD-CN): see switch_controlnet_tensors
        if not self.enable_controlnet or self.controlnet_script is None: return
"""),
        Block(
            "MD-CN tile memo",
            r"""    @controlnet
    def switch_controlnet_tensors(self, batch_id:int, x_batch_size:int, tile_batch_size:int, is_denoise=False):
        if not self.enable_controlnet: return
        if self.control_tensor_batch is None: return

        for param_id in range(len(self.control_params)):
            control_tile = self.control_tensor_batch[param_id][batch_id]
            if self.is_kdiff:
                all_control_tile = []
                for i in range(tile_batch_size):
                    this_control_tile = [control_tile[i].unsqueeze(0)] * x_batch_size
                    all_control_tile.append(torch.cat(this_control_tile, dim=0))
                control_tile = torch.cat(all_control_tile, dim=0)                                           
            else:
                control_tile = control_tile.repeat([x_batch_size if is_denoise else x_batch_size * 2, 1, 1, 1])
            self.control_params[param_id].hint_cond = control_tile.to(devices.device)

    @controlnet
    def set_custom_controlnet_tensors(self, bbox_id:int, repeat_size:int):
        if not self.enable_controlnet: return
        if not len(self.control_tensor_custom): return
        
        for param_id in range(len(self.control_params)):
            control_tensor = self.control_tensor_custom[param_id][bbox_id].to(devices.device)
            self.control_params[param_id].hint_cond = control_tensor.repeat((repeat_size, 1, 1, 1))
""",
            r"""    @controlnet
    def switch_controlnet_tensors(self, batch_id:int, x_batch_size:int, tile_batch_size:int, is_denoise=False):
        if not self.enable_controlnet: return
        if self.control_tensor_batch is None: return

        for param_id in range(len(self.control_params)):
            key = (param_id, batch_id, x_batch_size, tile_batch_size, is_denoise)
            control_tile = self.control_tile_device.get(key)
            if control_tile is None:
                control_tile = self.control_tensor_batch[param_id][batch_id]
                if self.is_kdiff:
                    all_control_tile = []
                    for i in range(tile_batch_size):
                        this_control_tile = [control_tile[i].unsqueeze(0)] * x_batch_size
                        all_control_tile.append(torch.cat(this_control_tile, dim=0))
                    control_tile = torch.cat(all_control_tile, dim=0)
                else:
                    control_tile = control_tile.repeat([x_batch_size if is_denoise else x_batch_size * 2, 1, 1, 1])
                control_tile = control_tile.to(devices.device)
                self.remember_control_tile(key, control_tile)
            self.assign_hint_cond(param_id, control_tile)

    @controlnet
    def set_custom_controlnet_tensors(self, bbox_id:int, repeat_size:int):
        if not self.enable_controlnet: return
        if not len(self.control_tensor_custom): return

        for param_id in range(len(self.control_params)):
            key = (param_id, 'custom', bbox_id, repeat_size)
            control_tile = self.control_tile_device.get(key)
            if control_tile is None:
                control_tensor = self.control_tensor_custom[param_id][bbox_id].to(devices.device)
                control_tile = control_tensor.repeat((repeat_size, 1, 1, 1))
                self.remember_control_tile(key, control_tile)
            self.assign_hint_cond(param_id, control_tile)

    # gb10 (MD-CN): every hint_cond assignment makes ControlNet drop the hint's derived state; colorfix, inpaint_only
    # and reference units then re-derive the hint latent, hashing the whole hint (host copy + sha1) per tile batch
    # per step. The device tile of one (unit, batch, batch shape) is built once per request, and a unit whose hint
    # already is that tile is not reassigned. Same tensor values: bit-identical. With "Move ControlNet tensor to CPU"
    # the tiles are still built per call, as before, so none is held on the device.
    def remember_control_tile(self, key, control_tile:Tensor):
        if not self.control_tensor_cpu:
            self.control_tile_device[key] = control_tile

    def assign_hint_cond(self, param_id:int, control_tile:Tensor):
        if self.control_params[param_id].hint_cond is not control_tile:
            self.control_params[param_id].hint_cond = control_tile
""",
            sentinel="    def assign_hint_cond(",
        ),
    ],
    "scripts/tilediffusion.py": [
        Block(
            "MD-NI-CACHE request scope and MD-REGION-COND",
            r"""        # unhijack & unhook, in case it broke at last time
        self.reset()

        if not enabled: return
""",
            r"""        # unhijack & unhook, in case it broke at last time
        self.reset()
        # gb10 (MD-NI-CACHE): the noise inversion cache never outlives a request (see AbstractDiffusion.sample_img2img).
        self.noise_inverse_cache = None

        if not enabled: return

        # gb10 (MD-REGION-COND): region prompts build tensor conditioning; with SDXL/SD3 dict conditioning the first
        # region forward raised a TypeError (torch.cat of dicts) only after sampling had started. Same region filter as
        # AbstractDiffusion.init_custom_bbox.
        if enable_bbox_control and (getattr(shared.sd_model, 'is_sdxl', False) or getattr(shared.sd_model, 'is_sd3', False)):
            if any(s.enable and s.x <= 1.0 and s.y <= 1.0 and s.w > 0.0 and s.h > 0.0 for s in build_bbox_settings(bbox_control_states).values()):
                raise RuntimeError('[Tiled Diffusion] Region prompt control supports SD1/SD2 models only; disable the regions for SDXL/SD3.')
""",
        ),
        Block("MD-NI-CACHE key from p", r"""    def noise_inverse_set_cache(self, p: ProcessingImg2Img, x0: Tensor, xt: Tensor, prompts: List[str], steps: int, retouch:float):
        self.noise_inverse_cache = NoiseInverseCache(p.sd_model.sd_model_hash, x0,  xt, steps, retouch, prompts)
""", r"""    def noise_inverse_set_cache(self, p: ProcessingImg2Img, x0: Tensor, xt: Tensor, prompts: List[str], steps: int, retouch:float):
        self.noise_inverse_cache = NoiseInverseCache(p.sd_model.sd_model_hash, x0,  xt, steps, retouch, prompts, p.extra_network_data)  # gb10: MD-NI-CACHE
"""),
    ],
    # DemoFusion shares AbstractDiffusion.sample_img2img and the NoiseInverseCache type: the same key and request scope.
    "scripts/tileglobal.py": [
        Block(
            "MD-NI-CACHE DemoFusion request scope",
            r"""        # unhijack & unhook, in case it broke at last time
        self.reset()
        p.mixture = mixture_mode
""",
            r"""        # unhijack & unhook, in case it broke at last time
        self.reset()
        # gb10 (MD-NI-CACHE): the noise inversion cache never outlives a request (see AbstractDiffusion.sample_img2img).
        self.noise_inverse_cache = None
        p.mixture = mixture_mode
""",
        ),
        Block("MD-NI-CACHE DemoFusion key from p", r"""    def noise_inverse_set_cache(self, p: ProcessingImg2Img, x0: Tensor, xt: Tensor, prompts: List[str], steps: int, retouch:float):
        self.noise_inverse_cache = NoiseInverseCache(p.sd_model.sd_model_hash, x0,  xt, steps, retouch, prompts)
""", r"""    def noise_inverse_set_cache(self, p: ProcessingImg2Img, x0: Tensor, xt: Tensor, prompts: List[str], steps: int, retouch:float):
        self.noise_inverse_cache = NoiseInverseCache(p.sd_model.sd_model_hash, x0,  xt, steps, retouch, prompts, p.extra_network_data)  # gb10: MD-NI-CACHE
"""),
    ],
    "tile_methods/mixtureofdiffusers.py": [
        Block(
            "MD-W1 precompute",
            r"""                self.custom_weights[bbox_id] *= self.rescale_factor[bbox.slicer]

    @grid_bbox
    def get_tile_weights(self) -> Tensor:
""",
            r"""                self.custom_weights[bbox_id] *= self.rescale_factor[bbox.slicer]
        # gb10 (MD-W1): the grid tile weights are fixed for the request, so build them once here (one tile_h x tile_w
        # fp32 map per grid tile) instead of once per tile per step. Same operands and op: bit-identical.
        self.batched_tile_weights = [[self.tile_weights * self.rescale_factor[bbox.slicer] for bbox in bboxes] for bboxes in self.batched_bboxes]

    @grid_bbox
    def get_tile_weights(self) -> Tensor:
""",
        ),
        Block(
            "MD-W1 use",
            # The upstream comment line ends with a space.
            "                for i, bbox in enumerate(bboxes):\n"
            "                    # This weights can be calcluated in advance, but will cost a lot of vram \n"
            "                    # when you have many tiles. So we calculate it here.\n"
            "                    w = self.tile_weights * self.rescale_factor[bbox.slicer]\n"
            "                    self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :] * w\n",
            r"""                for i, bbox in enumerate(bboxes):
                    self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :] * self.batched_tile_weights[batch_id][i]
""",
        ),
    ],
}


# The deploy10 release of the blocks that changed since (gb10/patchlib.py `previous`): installed copies patched by it
# are reverted to upstream and patched again.
PREVIOUS: dict[str, list[Block]] = {
    "tile_utils/utils.py": [
        Block(
            "MD terminal tile origins (deploy10)",
            BLOCKS["tile_utils/utils.py"][0].original,
            '''def _gb10_terminal_tile_origins(extent:int, tile:int, overlap:int) -> List[int]:
    if extent <= 0 or tile <= 0:
        raise ValueError(f"extent and tile must be positive, got extent={extent}, tile={tile}")
    terminal = max(0, extent - tile)
    if terminal == 0:
        return [0]
    stride = max(1, tile - overlap)
    origins = list(range(0, terminal + 1, stride))
    if origins[-1] != terminal:
        origins.append(terminal)
    return origins


def split_bboxes(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    x_origins = _gb10_terminal_tile_origins(w, tile_w, overlap)
    y_origins = _gb10_terminal_tile_origins(h, tile_h, overlap)

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=devices.device, dtype=torch.float32)
    for y in y_origins:
        for x in x_origins:
            bbox = BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight
''',
            sentinel="def _gb10_terminal_tile_origins",
        ),
    ],
    TILEVAE: BLOCKS[TILEVAE][:TILEVAE_DEPLOY10_BLOCKS],
}


def main() -> int:
    args = parse_cli(__doc__.splitlines()[0])
    targets = {args.path / relative: blocks for relative, blocks in BLOCKS.items()}
    previous = {args.path / relative: blocks for relative, blocks in PREVIOUS.items()}
    written = apply_blocks(targets, label=LABEL, check=args.check, previous=previous)
    for path in written:
        print(f"Patched MultiDiffusion changes: {path}")
    if not written:
        print(f"MultiDiffusion changes {'verified' if args.check else 'already patched'}: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
