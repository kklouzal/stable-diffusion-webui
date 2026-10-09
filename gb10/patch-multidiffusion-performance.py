#!/usr/bin/env python3
"""Patch and verify the GB10 changes to the mounted MultiDiffusion / Tiled VAE extension.

Each change is an exact ORIGINAL -> PATCHED text block (gb10/patchlib.py: a target is either fully original and gets
patched, or fully patched and is left alone; anything else, CRLF included, aborts the deploy; nothing is written until
every target verifies; --check writes nothing). ORIGINAL is upstream origin/main 22798f6, so a fresh install patches
cleanly. The blocks:
- MD terminal tile origins (tile_utils/utils.py): upstream spreads origins as int(col * (w - tile_w) / (cols - 1));
  the float floor can leave the last latent column or row uncovered (weight 0). The replacement steps by
  tile - overlap and appends the terminal origin. Its tile count equals upstream's, but the origins differ from
  upstream for most extents, not only for the uncovered ones.
- TV-FB: the former local commit "Modernize Tiled VAE attention fallbacks" (5022f68): xformers is optional and falls
  back to SDP, and the sage2/sage3 method names map to SDP. Its torch.nn.attention import is no longer used since
  TV-ATTN; it is kept, like the sage names, so the deployed bytes stay unchanged.
- TV-*, MD-W1: performance changes. Exactness of each is argued next to the code it patches and tested on CPU against
  the unpatched extension (test/test_gb10_multidiffusion_performance_patcher.py).
A checkout holding only the former 0001 commit (upstream + TV-FB, nothing else) is not accepted: reset it to upstream.
"""
from __future__ import annotations

from patchlib import Block, apply_blocks, parse_cli

LABEL = "MultiDiffusion"
TILEVAE = "scripts/tilevae.py"

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


def main() -> int:
    args = parse_cli(__doc__.splitlines()[0])
    targets = {args.path / relative: blocks for relative, blocks in BLOCKS.items()}
    written = apply_blocks(targets, label=LABEL, check=args.check)
    for path in written:
        print(f"Patched MultiDiffusion changes: {path}")
    if not written:
        print(f"MultiDiffusion changes {'verified' if args.check else 'already patched'}: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
