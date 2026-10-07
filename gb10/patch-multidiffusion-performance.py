#!/usr/bin/env python3
"""Patch and verify the GB10 performance changes to the mounted MultiDiffusion / Tiled VAE extension.

Each change is an exact ORIGINAL -> PATCHED text block. ORIGINAL blocks match the host checkout as deployed: upstream
plus the 0001-modern-attention-fallbacks commit (5022f68) and patch-multidiffusion-terminal-tiles.py.
- A target file must be either fully original (it gets patched) or fully patched (it is left alone), or hold an
  earlier version of some patched blocks (SUPERSEDED) with every other block patched (those blocks are upgraded).
- Anything else aborts the deploy: unknown upstream text, a partial patch, CRLF line endings.
- Every target is validated before any file is written. Each file is replaced atomically, keeping its mode and owner.
- --check writes nothing and fails unless every target is fully patched.

Exactness of each change is argued next to the code it patches and tested on CPU against the unpatched extension
(tests/test_gb10_multidiffusion_performance_patcher.py).
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

TILEVAE = "scripts/tilevae.py"

# tile_utils/attn.py TV-ATTN blocks. *_V1 is the text deployed before the upcast/autocast fix: a file holding it is
# upgraded in place (see SUPERSEDED), never treated as unknown drift.
TV_ATTN_IMPORT_V1 = 'from modules.sd_hijack_optimizations import get_available_vram, get_xformers_flash_attention_op, run_scaled_dot_product_attention, sub_quad_attention\n'
TV_ATTN_IMPORT = 'from modules.devices import without_autocast\n' + TV_ATTN_IMPORT_V1.replace('sub_quad_attention\n', 'sub_quad_attention  # gb10: TV-ATTN\n')
TV_ATTN_SDPA_V1 = r"""def sdp_no_mem_attnblock_forward(self, x):
    return sdp_attnblock_forward(self, x, sdpa_backend_override="flash,math")

def sdp_attnblock_forward(self, h_, sdpa_backend_override=None):
    # gb10 (TV-ATTN): one head of 4-D [b, 1, hw, c] q/k/v. PyTorch's fused SDPA kernels need 4-D inputs, so the
    # old 3-D [b, hw, c] call always fell back to the math kernel and materialized the hw x hw scores (about 12 GB
    # bf16 for one 278x278 decoder tile). webui's helper also applies its SDPA backend policy. Same attention,
    # different valid kernel: numerically equivalent.
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = (t.reshape(b, 1, c, h * w).transpose(-1, -2) for t in (q, k, v))
    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = run_scaled_dot_product_attention(q, k, v, is_causal=False, sdpa_backend_override=sdpa_backend_override)
    out = out.to(dtype)
    out = out.transpose(-1, -2).reshape(b, c, h, w)
    out = self.proj_out(out)
    return out
"""
TV_ATTN_SDPA = r"""def sdp_no_mem_attnblock_forward(self, x):
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
"""


BLOCKS: dict[str, list[tuple[str, str, str]]] = {
    TILEVAE: [
        (
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
        (
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
        (
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
        (
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
        (
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
        (
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
        (
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
        (
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
        (
            "TV-ATTN import",
            r"""from modules.sd_hijack_optimizations import get_available_vram, get_xformers_flash_attention_op, sub_quad_attention
""",
            TV_ATTN_IMPORT,
        ),
        (
            "TV-ATTN 4-D SDPA",
            r"""def sdp_no_mem_attnblock_forward(self, x):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
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
            TV_ATTN_SDPA,
        ),
    ],
    "tile_methods/mixtureofdiffusers.py": [
        (
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
        (
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


# Earlier PATCHED texts of a block, keyed by (target, block name). A deployed file holding them is upgraded to the
# current PATCHED text; --check fails until it is.
SUPERSEDED: dict[tuple[str, str], tuple[str, ...]] = {
    ("tile_utils/attn.py", "TV-ATTN import"): (TV_ATTN_IMPORT_V1,),
    ("tile_utils/attn.py", "TV-ATTN 4-D SDPA"): (TV_ATTN_SDPA_V1,),
}


def block_states(relative: str, source: str) -> list[tuple[str, str, str]]:
    """(state, current text, patched text) per block: state is "original", "patched" or "superseded".

    Exactly one of the block's known texts must occur, exactly once; anything else raises SystemExit."""
    if "\r" in source:
        raise SystemExit(f"unsupported MultiDiffusion source (CRLF line endings): {relative}")
    states = []
    for name, original, patched in BLOCKS[relative]:
        known = [("original", original), ("patched", patched)] + [("superseded", text) for text in SUPERSEDED.get((relative, name), ())]
        counts = [source.count(text) for _state, text in known]
        found = [entry for entry, count in zip(known, counts) if count]
        if len(found) != 1 or sum(counts) != 1:
            raise SystemExit(f"unsupported MultiDiffusion source for {name} (original/patched/superseded x{counts}): {relative}")
        states.append((found[0][0], found[0][1], patched))
    return states


def file_state(relative: str, source: str) -> str:
    """Return "original", "patched" or "superseded" (every block patched or superseded, at least one superseded)
    for a target file's text, or raise SystemExit for anything else (unknown text, a partial patch, CRLF)."""
    states = {state for state, _current, _patched in block_states(relative, source)}
    if states == {"original"} or states == {"patched"}:
        return states.pop()
    if "original" not in states:
        return "superseded"
    raise SystemExit(f"partially patched MultiDiffusion source: {relative}")


def replace_atomically(path: Path, text: str) -> None:
    stat = path.stat()
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".gb10-tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.chmod(tmp_name, stat.st_mode & 0o7777)
        if os.geteuid() == 0:
            os.chown(tmp_name, stat.st_uid, stat.st_gid)
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="multidiffusion-upscaler-for-automatic1111 checkout")
    parser.add_argument("--check", action="store_true", help="verify that every target is fully patched; write nothing")
    args = parser.parse_args()

    pending: dict[Path, str] = {}
    for relative, blocks in BLOCKS.items():
        path = args.root / relative
        if not path.is_file():
            raise SystemExit(f"MultiDiffusion source not found: {path}")
        source = path.read_bytes().decode("utf-8")
        state = file_state(relative, source)
        if args.check and state != "patched":
            raise SystemExit(f"MultiDiffusion performance patch {'outdated' if state == 'superseded' else 'missing'}: {path}")
        if state != "patched":
            for _state, current, patched in block_states(relative, source):
                source = source.replace(current, patched, 1)
            pending[path] = source

    for path, text in pending.items():
        replace_atomically(path, text)
        print(f"Patched MultiDiffusion performance changes: {path}")
    if not pending:
        print(f"MultiDiffusion performance changes {'verified' if args.check else 'already patched'}: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
