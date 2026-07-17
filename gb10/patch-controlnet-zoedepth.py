#!/usr/bin/env python3
"""Apply the GB10 ZoeDepth/timm checkpoint compatibility guard."""
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
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
    print(f"ZoeDepth compatibility patch already present: {path}")
elif old in source:
    path.write_text(source.replace(old, new, 1))
    print(f"Patched ZoeDepth checkpoint compatibility: {path}")
else:
    raise SystemExit(f"unsupported ZoeDepth loader implementation: {path}")
