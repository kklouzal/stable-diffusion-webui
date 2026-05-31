# GB10 mounted extension audit

This document records the current external A1111 extension posture now that A1111-Controller is canonical for Schwi's workflow direction.

The goal is to keep the runtime lean: purge UI-only extensions that A1111-Controller supersedes, and adopt any remaining generation-critical behavior as first-class repo-owned source instead of depending on opaque mounted checkouts.

## Current live extension inventory

Live host path:

- `/opt/gb10/stable-diffusion/Extensions`

Expected retained directories after the 2026-05-03 cleanup pass:

- `multidiffusion-upscaler-for-automatic1111`
- `openclaw-clear-cond-cache`
- `sd-webui-detail-daemon`
- `sd-webui-incantations`
- `sd-webui-model-converter`
- `ultimate-upscale-for-automatic1111`

## Ownership policy

### First-class extensions

First-class means:

- source lives in this repo under `extensions/`
- provenance/license is documented
- `gb10/run.sh` syncs the repo-owned source into the host `Extensions/` mount before launch
- behavior changes are committed, reviewable, and validated with the image/runtime

Currently first-class:

- `openclaw-clear-cond-cache`
  - owns OpenClaw/A1111-Controller helper endpoints:
    - `POST /sdapi/v1/openclaw/clear-cond-cache`
    - `GET /sdapi/v1/openclaw/cond-cache`
    - `POST /sdapi/v1/openclaw/token-count`
    - `POST /sdapi/v1/openclaw/token_counter`
  - replaces the former token-counter dependency on `sd-webui-prompt-all-in-one`
- `sd-webui-incantations`
  - owns PAG, SEG, CFG-combiner, and Dynamic Thresholding / CFG-Fix behavior
  - replaces previous dependence on separate Incantations and Dynamic Thresholding checkouts

### External mounted extensions

External mounted extensions are tolerated only if they provide behavior we still need and are not yet worth adopting. Any external extension that materially affects generation quality, callback ordering, model loading, or high-value workflow behavior should either become first-class or be removed.

## Removal decisions

Schwi approved removing these UI-only / Controller-superseded extensions from the live A1111 runtime:

- `Config-Presets`
- `model-keyword`
- `sd_delete_button`
- `sd-webui-cardmaster`
- `sd-webui-prompt-all-in-one`
- `sd-webui-state-manager`

`sd-webui-state-manager` was already disabled in `config.json`. The user referred to this as `sd-webui-statemaster`; the live directory name is `sd-webui-state-manager`.

`sd-webui-prompt-all-in-one` originally stayed because A1111-Controller used its `/physton_prompt/token_counter` endpoint. That dependency was replaced by the first-class `openclaw-clear-cond-cache` endpoint `/sdapi/v1/openclaw/token-count`, after which the controller fallback to `/physton_prompt/token_counter` was removed.

Removal result:

- earlier UI-only removals were quarantined under `/opt/gb10/stable-diffusion/Extensions.quarantine/20260503-160304`
- the quarantine tree `/opt/gb10/stable-diffusion/Extensions.quarantine` was purged completely after Schwi validated the runtime
- `sd-webui-prompt-all-in-one` was removed after owned token-count replacement and live validation
- A1111 restarted successfully from `local/gb10-a1111:base-protected-app-latest`
- smoke tests should include progress endpoint health, model listing, OpenClaw token-count endpoint, Controller token-count endpoint, and extension absence

## Keep decisions

Schwi confirmed these external mounted extensions need to stay because A1111-Controller uses functionality from them:

### `sd-webui-model-converter`

Decision: **keep**.

Reason:

- A1111-Controller uses model-conversion functionality.
- It should remain mounted for now.

Future ownership:

- consider adopting first-class or replacing with a Controller/offline utility only after identifying the exact conversion operations Controller depends on.

### `sd-webui-detail-daemon`

Decision: **keep**.

Reason:

- A1111-Controller uses this generation-control functionality.
- It modifies generation behavior through sampling/noise scheduling, so if we patch it later, it should be treated as output-quality-affecting code.

Future ownership:

- likely first-class adoption candidate if it remains central to workflows.

### `multidiffusion-upscaler-for-automatic1111`

Decision: **keep**.

Reason:

- A1111-Controller uses tiled diffusion / tiled VAE / large-image functionality from it.
- It already has GB10-specific xformers/SDPA compatibility handling.

Future ownership:

- strong first-class adoption candidate, because patched generation/runtime behavior should not remain opaque long-term.

Current repo note:

- `patches/mounted-extensions/multidiffusion-upscaler-for-automatic1111/0001-modern-attention-fallbacks.patch` records the local attention fallback patch.

### `ultimate-upscale-for-automatic1111`

Decision: **keep**.

Reason:

- A1111-Controller uses this upscaling functionality.

Future ownership:

- possible first-class adoption or replacement candidate after mapping exactly which upscale path Controller calls.

## Current retained external mounted extensions

After the approved removal/purge pass, the live external set is:

- `multidiffusion-upscaler-for-automatic1111`
- `sd-webui-detail-daemon`
- `sd-webui-model-converter`
- `ultimate-upscale-for-automatic1111`

Already first-class / keep:

- `openclaw-clear-cond-cache`
- `sd-webui-incantations`

## Proposed adoption order

1. `multidiffusion-upscaler-for-automatic1111`
   - already locally patched for GB10 attention behavior
   - generation/runtime hot path
2. `sd-webui-detail-daemon`
   - generation-affecting sampling/noise behavior
3. `ultimate-upscale-for-automatic1111`
   - high-value upscale workflow if Controller relies on it
4. `sd-webui-model-converter`
   - possibly better as an offline/Controller utility than an always-mounted A1111 extension

## Cleanup boundary

Future extension removals should still be done as deliberate remove/restart/smoke passes. The current approved removal set has been purged and validated.
