#!/usr/bin/env python3
"""Patch and verify the Ultimate Upscale run lifecycle: shared state, override settings and the reported result.

Exact ORIGINAL -> PATCHED text blocks on gb10/patchlib.py's contract; ORIGINAL is upstream Coyote-A master 2322caa.
- USDUpscaler.process() called state.begin() first and state.end() last. The request's job already owns the shared
  state (modules/api/api.py begins and ends it around the script), so the nested begin cleared an interrupt sent
  while the upscaler ran (every tile then ran and the request reported success), emptied the device cache once more,
  and renamed the job; the nested end emptied it again. process() no longer touches the job state; an exception
  leaves it to the caller's finally.
- override_settings are applied once around all tiles and restored afterwards (process_images() applies and restores
  them per tile, and an sd_vae override reloaded the VAE twice per tile). Each tile's process_images() then finds
  them already applied: same options for every tile, so the same images.
- A redraw pass that ran no tile (interrupted) raised UnboundLocalError on `processed`; a seams pass that ran no tile
  (a single tile row or column, or interrupted) blanked the infotext and appended the unchanged image a second
  time; the intersections pass dropped the half-tile pass's infotext when it ran no corner tile.
The deploy10 release (the former AST re-indent, marker OPENCLAW_ULTIMATE_UPSCALE_STATE_FINALLY_V2) is upgraded.
Runs before patch-ultimate-upscale-subcanvas.py; neither patcher's blocks overlap the other's.
"""
from __future__ import annotations

from pathlib import Path

from patchlib import Block, apply_blocks, parse_cli

LABEL = "Ultimate Upscale lifecycle"
TARGET_RELATIVE = Path("scripts") / "ultimate-upscale.py"
MARKER = "OPENCLAW_ULTIMATE_UPSCALE_LIFECYCLE_V3"

PROCESS = '''    def process(self):
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

BLOCKS = [
    Block(
        "process",
        PROCESS,
        f'''    def process(self):
        # {MARKER} (gb10/patch-ultimate-upscale-state-lifecycle.py): the request's job owns shared.state, so no
        # nested state.begin()/end() (it cleared an interrupt sent during the upscaler). override_settings apply once
        # around all tiles instead of being applied and restored (an sd_vae override reloaded) per tile.
        self.calc_jobs_count()
        self.result_images = []
        stored_opts = processing.store_processing_override_settings(self.p)
        try:
            processing.apply_processing_override_settings(self.p)
            if self.redraw.enabled:
                self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
                self.initial_info = self.redraw.initial_info
            self.result_images.append(self.image)
            if self.redraw.save:
                self.save_image()

            if self.seams_fix.enabled:
                image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
                # A seams pass that ran no tile returns its input unchanged and leaves initial_info None.
                if self.seams_fix.initial_info is not None:
                    self.image = image
                    self.initial_info = self.seams_fix.initial_info
                    self.result_images.append(self.image)
                    if self.seams_fix.save:
                        self.save_image()
        finally:
            if self.p.override_settings_restore_afterwards:
                processing.restore_processing_override_settings(stored_opts)
''',
        sentinel=MARKER,
    ),
    Block(
        "linear redraw without tiles",
        "        mask, draw = self.init_draw(p, image.width, image.height)\n        for yi in range(rows):\n",
        "        mask, draw = self.init_draw(p, image.width, image.height)\n        processed = None\n        for yi in range(rows):\n",
    ),
    Block(
        "chess redraw without tiles",
        "        mask, draw = self.init_draw(p, image.width, image.height)\n        tiles = []\n",
        "        mask, draw = self.init_draw(p, image.width, image.height)\n        processed = None\n        tiles = []\n",
    ),
    Block(
        "redraw infotext without tiles",
        "\n        p.width = image.width\n        p.height = image.height\n        self.initial_info = processed.infotext(p, 0)\n",
        "\n        p.width = image.width\n        p.height = image.height\n        if processed is not None:  # None: interrupted before the first tile\n"
        "            self.initial_info = processed.infotext(p, 0)\n",
        count=2,
    ),
    Block(
        "intersections keep the half-tile infotext",
        "        fixed_image = self.half_tile_process(p, image, rows, cols)\n        processed = None\n        self.init_draw(p)\n",
        "        fixed_image = self.half_tile_process(p, image, rows, cols)\n        processed = None\n"
        "        half_tile_info = self.initial_info\n        self.init_draw(p)\n"
        "        self.initial_info = half_tile_info  # kept when no intersection tile runs\n",
    ),
]

# deploy10: the former AST patch re-indented process() into try/finally around the nested begin/end.
DEPLOY10 = [Block("process", PROCESS, '''    def process(self):
        state.begin()
        try:
            # OPENCLAW_ULTIMATE_UPSCALE_STATE_FINALLY_V2: one begin owns one end.
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
''')]


def main() -> int:
    args = parse_cli(__doc__.splitlines()[0])
    target = args.path / TARGET_RELATIVE if args.path.is_dir() else args.path
    if apply_blocks({target: BLOCKS}, label=LABEL, check=args.check, previous={target: DEPLOY10}):
        print(f"Patched Ultimate Upscale state lifecycle: {target}")
    else:
        print(f"Ultimate Upscale lifecycle verified: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
