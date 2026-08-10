#!/usr/bin/env python3
"""Patch MultiDiffusion tile origins to include terminal latent edges."""
from __future__ import annotations

from pathlib import Path
import sys

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


def patch_file(path: Path) -> bool:
    if not path.exists():
        raise SystemExit(f"MultiDiffusion source not found: {path}")
    if path.name != "utils.py" or path.parent.name != "tile_utils":
        raise SystemExit(f"refusing unexpected MultiDiffusion target (expected tile_utils/utils.py): {path}")
    source = path.read_text(encoding="utf-8")
    if PATCHED in source:
        if ORIGINAL in source:
            raise SystemExit(f"ambiguous MultiDiffusion source contains original and patched blocks: {path}")
        return False
    if source.count(ORIGINAL) != 1:
        raise SystemExit(f"unsupported MultiDiffusion split_bboxes implementation: {path}")
    path.write_text(source.replace(ORIGINAL, PATCHED, 1), encoding="utf-8")
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
