#!/usr/bin/env python3
"""Patch Ultimate Upscale state lifecycle so state.end() runs exactly once."""
from __future__ import annotations

from pathlib import Path
import sys

TARGET_RELATIVE = Path("scripts") / "ultimate-upscale.py"

ORIGINAL = '''    def process(self):
        state.begin()
        self.calc_jobs_count()
        self.result_images = []
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)
        if self.redraw.save:
            self.save_image()

        if self.seams_fix.enabled:
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
            if self.seams_fix.save:
                self.save_image()
        state.end()
'''

PATCHED = '''    def process(self):
        state.begin()
        try:
            self.calc_jobs_count()
            self.result_images = []
            if self.redraw.enabled:
                self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
                self.initial_info = self.redraw.initial_info
            self.result_images.append(self.image)
            if self.redraw.save:
                self.save_image()

            if self.seams_fix.enabled:
                self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
                self.initial_info = self.seams_fix.initial_info
                self.result_images.append(self.image)
                if self.seams_fix.save:
                    self.save_image()
        finally:
            state.end()
'''


def resolve_target(arg: str) -> Path:
    path = Path(arg)
    if path.is_dir():
        path = path / TARGET_RELATIVE
    return path


def patch_file(path: Path) -> bool:
    if not path.exists():
        raise SystemExit(f"Ultimate Upscale source not found: {path}")
    if path.name != "ultimate-upscale.py" or path.parent.name != "scripts":
        raise SystemExit(f"refusing unexpected Ultimate Upscale target (expected scripts/ultimate-upscale.py): {path}")
    source = path.read_text(encoding="utf-8")
    if PATCHED in source:
        if ORIGINAL in source:
            raise SystemExit(f"ambiguous Ultimate Upscale source contains original and patched blocks: {path}")
        return False
    if source.count(ORIGINAL) != 1:
        raise SystemExit(f"unsupported Ultimate Upscale process lifecycle implementation: {path}")
    path.write_text(source.replace(ORIGINAL, PATCHED, 1), encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} /path/to/ultimate-upscale-for-automatic1111-or-ultimate-upscale.py", file=sys.stderr)
        return 2
    target = resolve_target(argv[1])
    changed = patch_file(target)
    if changed:
        print(f"Patched Ultimate Upscale state lifecycle: {target}")
    else:
        print(f"Ultimate Upscale state lifecycle already patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
