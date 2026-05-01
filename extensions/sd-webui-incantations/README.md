# GB10 Incantations Extension

This is the GB10-owned vendored guidance extension, combining Incantations PAG/SEG/CFG-combiner behavior with Dynamic Thresholding / CFG-Fix for the GB10 A1111 image.

## Provenance

### Dynamic Thresholding / CFG-Fix provenance

- Upstream: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding
- Vendored from local upstream checkout commit: `73e4e04565aa86237d66764ac58ffae1f7e40e48`
- License: MIT, preserved in `DYNAMIC-THRESHOLDING-LICENSE.txt`
- Note: the combined extension as a whole remains GPL-3.0-compatible because the Incantations code is GPL-3.0; Dynamic Thresholding sources retain their MIT notice.
- Upstream documentation was intentionally not retained as shipped docs because this repo now owns a trimmed A1111-only integration.

### Incantations provenance

- Upstream: https://github.com/v0xie/sd-webui-incantations
- Vendored from local upstream-derived checkout commit: `769a55cade195c3c0718c41930adff9865052aac`
- Local source branch at adoption time: `gb10-local`
- License: GPL-3.0, preserved in `LICENSE`
- Upstream documentation was intentionally not retained as shipped docs because removed legacy features would make it misleading.

## Source map

- `scripts/dynamic_thresholding.py`, `dynthres_core.py`, and `dynthres_unipc.py`: A1111 Dynamic Thresholding / CFG-Fix source. ComfyUI/SwarmUI entrypoints from the old standalone extension were removed.
- `scripts/pag.py` and `scripts/smoothed_energy_guidance.py`: Incantations guidance source with GB10 lifecycle fixes.
- `scripts/cfg_combiner.py`: GB10-owned CFG composition glue for PAG, CFG interval scheduling, and CFG-Fix coexistence.
- Removed abandoned A1111-discovered Incantations scripts: legacy prompt incanting, S-CFG, T2I-Zero, and attention-map saving. They were not part of the GB10 active guidance path and still used stale callback cleanup / debug code.
- `scripts/incantation_base.py`: GB10-trimmed A1111 entrypoint that exposes only the supported combined guidance stack.


## A1111 API argument order

Keep the `incantations` always-on script argument order stable unless a caller migration is planned. The UI labels and `elem_id`s may be clarified, but positional API callers depend on this order:

1. `seg_active`
2. `seg_blur_sigma`
3. `seg_start_step`
4. `seg_end_step`
5. `pag_active`
6. `pag_scale`
7. `pag_start_step`
8. `pag_end_step`
9. `cfg_interval_enable`
10. `cfg_interval_schedule`
11. `cfg_interval_low`
12. `cfg_interval_high`
13. `pag_sanf`

This ordering intentionally differs from the PAG UI visual order because `pag_sanf` was appended last historically. Preserve behavior over cosmetic ordering.

## GB10 ownership doctrine

The upstream extension appears effectively abandoned for our runtime needs, while PAG/SEG/CFG-combiner and CFG-Fix behavior are now quality-critical for this image. Treat this directory as first-class GB10 source:

- do not depend on a live external extension checkout for runtime behavior
- do not load a separate `sd-dynamic-thresholding` extension alongside this combined owned extension
- do not restore old upstream README/ComfyUI/SwarmUI/package-manager surfaces unless this repo actually supports them again
- keep local changes reviewable in this repository
- preserve GPL-3.0 notices and upstream provenance
- prefer conservative, testable fixes before changing guidance math
- audit hook cleanup and CFG-combiner interactions carefully because stale hooks or wrapper-order drift can materially affect image quality
- keep `scripts/cfg_combiner.py` compatible with CFG-Fix / Dynamic Thresholding by delegating the base CFG result to the captured original `combine_denoised` callable before adding PAG guidance
- avoid reintroducing abandoned patch-stack cleanup machinery in PAG/SEG; hook cleanup should be explicit and local
- do not add experimental scripts under `scripts/` unless they are actively maintained; A1111 auto-discovers them
