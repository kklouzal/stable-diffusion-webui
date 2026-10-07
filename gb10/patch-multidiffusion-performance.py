#!/usr/bin/env python3
"""Patch and verify the GB10 performance changes to the mounted MultiDiffusion / Tiled VAE extension.

Each change is an exact ORIGINAL -> PATCHED text block. ORIGINAL blocks match the host checkout as deployed: upstream
plus the 0001-modern-attention-fallbacks commit (5022f68) and patch-multidiffusion-terminal-tiles.py.
- A target file must be either fully original (it gets patched) or fully patched (it is left alone).
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
}


def file_state(relative: str, source: str) -> str:
    """Return "original" or "patched" for a target file's text, or raise SystemExit for anything else."""
    if "\r" in source:
        raise SystemExit(f"unsupported MultiDiffusion source (CRLF line endings): {relative}")
    states = []
    for name, original, patched in BLOCKS[relative]:
        counts = (source.count(original), source.count(patched))
        if counts == (1, 0):
            states.append("original")
        elif counts == (0, 1):
            states.append("patched")
        else:
            raise SystemExit(f"unsupported MultiDiffusion source for {name} (original x{counts[0]}, patched x{counts[1]}): {relative}")
    if len(set(states)) != 1:
        raise SystemExit(f"partially patched MultiDiffusion source: {relative}")
    return states[0]


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
            raise SystemExit(f"MultiDiffusion performance patch missing: {path}")
        if state == "original":
            for _name, original, patched in blocks:
                source = source.replace(original, patched, 1)
            pending[path] = source

    for path, text in pending.items():
        replace_atomically(path, text)
        print(f"Patched MultiDiffusion performance changes: {path}")
    if not pending:
        print(f"MultiDiffusion performance changes {'verified' if args.check else 'already patched'}: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
