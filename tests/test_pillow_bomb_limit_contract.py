"""Nothing in the project disables Pillow's decompression-bomb check (Image.MAX_IMAGE_PIXELS = None).

The limit is process-wide: one X/Y/Z plot used to switch it off for every later API image decode.
"""

import ast
from pathlib import Path

ROOTS = ("modules", "scripts", "extensions-builtin", "extensions")


def _assignments_to_max_image_pixels():
    for root in ROOTS:
        for path in sorted(Path(root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(isinstance(target, ast.Attribute) and target.attr == "MAX_IMAGE_PIXELS" for target in targets):
                        yield path, node


def test_no_code_disables_the_decompression_bomb_limit():
    offenders = [f"{path}:{node.lineno}" for path, node in _assignments_to_max_image_pixels()
                 if isinstance(node.value, ast.Constant) and node.value.value is None]
    assert offenders == []


def test_xyz_grid_only_raises_the_limit_to_the_configured_grid_size():
    sites = [(path.as_posix(), ast.unparse(node.value)) for path, node in _assignments_to_max_image_pixels()]
    assert sites == [("scripts/xyz_grid.py", "max(Image.MAX_IMAGE_PIXELS, int(opts.img_max_size_mp * 1000000))")]
