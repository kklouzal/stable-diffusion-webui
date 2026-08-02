#!/usr/bin/env python3
"""Apply GB10 ZoeDepth compatibility fixes for current torch/timm runtimes."""
from __future__ import annotations

from pathlib import Path
import sys


def patch_loader(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    old = "        model.load_state_dict(torch.load(modelpath, map_location=model.device)['model'])\n"
    new = '''        incompatible = model.load_state_dict(
            torch.load(modelpath, map_location=model.device)['model'], strict=False
        )
        unsupported_missing = list(incompatible.missing_keys)
        unsupported_unexpected = [
            key for key in incompatible.unexpected_keys
            if not key.endswith(".attn.relative_position_index")
        ]
        if unsupported_missing or unsupported_unexpected:
            raise RuntimeError(
                "Unsupported ZoeDepth checkpoint mismatch: "
                f"missing={unsupported_missing}, unexpected={unsupported_unexpected}"
            )
'''
    if new in source:
        return False
    if old not in source:
        raise SystemExit(f"unsupported ZoeDepth loader implementation: {path}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    return True


def patch_attractor(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    patched = source.replace("@torch.jit.script\ndef exp_attractor", "def exp_attractor")
    patched = patched.replace("@torch.jit.script\ndef inv_attractor", "def inv_attractor")
    if patched == source:
        if "@torch.jit.script" in source:
            raise SystemExit(f"unsupported ZoeDepth attractor torch.jit.script usage: {path}")
        return False
    path.write_text(patched, encoding="utf-8")
    return True


def patch_timm_import(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    old = "from timm.models.layers import get_act_layer"
    new = "from timm.layers import get_act_layer"
    if new in source:
        return False
    if old not in source:
        raise SystemExit(f"unsupported ZoeDepth timm import implementation: {path}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: patch-controlnet-zoedepth.py /path/to/sd-webui-controlnet/annotator/zoe/__init__.py")

    init_path = Path(argv[1]).resolve()
    if init_path.name != "__init__.py" or init_path.parent.name != "zoe":
        raise SystemExit(f"expected ControlNet annotator/zoe/__init__.py, got: {init_path}")

    zoe_root = init_path.parent
    targets = {
        "checkpoint loader": (patch_loader, init_path),
        "deprecated torch.jit.script attractors": (patch_attractor, zoe_root / "zoedepth" / "models" / "layers" / "attractor.py"),
        "deprecated timm.models.layers import": (patch_timm_import, zoe_root / "zoedepth" / "models" / "base_models" / "midas_repo" / "midas" / "dpt_depth.py"),
    }
    changed: list[str] = []
    for label, (func, target) in targets.items():
        if not target.exists():
            raise SystemExit(f"missing ZoeDepth {label} target: {target}")
        if func(target):
            changed.append(f"{label}: {target}")

    if changed:
        print("Patched ZoeDepth compatibility: " + "; ".join(changed))
    else:
        print(f"ZoeDepth compatibility patches already present: {zoe_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
