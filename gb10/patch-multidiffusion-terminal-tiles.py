#!/usr/bin/env python3
"""Patch MultiDiffusion tile origins to include terminal latent edges.

Upstream spreads origins as int(col * (w - tile_w) / (cols - 1)); the float floor can leave the last latent column
or row uncovered (weight 0). The replacement steps by tile - overlap and appends the terminal origin. Its tile
count equals upstream's, but the origins differ from upstream for most extents, not only for the uncovered ones.
The source must be UTF-8 with LF line endings; anything but exactly-original or exactly-patched fails closed.
The patched text is verified (one patched block, no original block, valid Python) before it is written, and the
file is replaced atomically, keeping its mode and owner.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

TARGET_RELATIVE = Path("tile_utils") / "utils.py"

ORIGINAL = '''def split_bboxes(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
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
'''

PATCHED = '''def _gb10_terminal_tile_origins(extent:int, tile:int, overlap:int) -> List[int]:
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
'''


def resolve_target(arg: str) -> Path:
    path = Path(arg)
    if path.is_dir():
        path = path / TARGET_RELATIVE
    return path


def verify(source: str, path: Path) -> None:
    """Post-condition of a patch: exactly one patched block and helper, no original block, valid Python."""
    if source.count(PATCHED) != 1 or ORIGINAL in source or source.count("def _gb10_terminal_tile_origins") != 1:
        raise SystemExit(f"MultiDiffusion terminal tiles verification failed (partial patch): {path}")
    try:
        compile(source, str(path), "exec", dont_inherit=True)
    except SyntaxError as exc:
        raise SystemExit(f"MultiDiffusion terminal tiles verification failed (invalid Python): {path}: {exc}") from exc


def replace_atomically(path: Path, text: str) -> None:
    """Write text (UTF-8, newlines untranslated) to a temporary file next to path, fsync it, give it path's mode
    (and owner when run as root, as run.sh does) and os.replace() path with it: a failure at any point leaves path
    untouched and removes the temporary file. A symlinked path is written through, as an in-place write would."""
    path = Path(os.path.realpath(path))
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
    except BaseException as exc:
        try:
            os.unlink(tmp_name)
        except OSError as cleanup:
            exc.add_note(f"could not remove the temporary file {tmp_name}: {cleanup}")
        raise


def patch_file(path: Path) -> bool:
    if not path.exists():
        raise SystemExit(f"MultiDiffusion source not found: {path}")
    if path.name != "utils.py" or path.parent.name != "tile_utils":
        raise SystemExit(f"refusing unexpected MultiDiffusion target (expected tile_utils/utils.py): {path}")
    source = path.read_bytes().decode("utf-8")  # no newline translation: CRLF must fail closed, not be rewritten
    if "\r" in source:
        raise SystemExit(f"unsupported MultiDiffusion source (CRLF line endings): {path}")
    if PATCHED in source:
        if ORIGINAL in source or source.count(PATCHED) != 1:
            raise SystemExit(f"ambiguous MultiDiffusion source contains original and patched blocks: {path}")
        return False
    if source.count(ORIGINAL) != 1 or "def _gb10_terminal_tile_origins" in source:
        raise SystemExit(f"unsupported MultiDiffusion split_bboxes implementation: {path}")
    patched = source.replace(ORIGINAL, PATCHED, 1)
    verify(patched, path)
    replace_atomically(path, patched)
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} /path/to/multidiffusion-upscaler-for-automatic1111-or-utils.py", file=sys.stderr)
        return 2
    target = resolve_target(argv[1])
    changed = patch_file(target)
    if changed:
        print(f"Patched MultiDiffusion terminal tile origins: {target}")
    else:
        print(f"MultiDiffusion terminal tile origins already patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
