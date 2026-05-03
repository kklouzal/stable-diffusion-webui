# GB10 mounted extension audit

This document records the current external A1111 extension posture now that A1111-Controller is canonical for Schwi's workflow direction.

The goal is to keep the runtime lean: purge UI-only extensions that A1111-Controller supersedes, and adopt any remaining generation-critical behavior as first-class repo-owned source instead of depending on opaque mounted checkouts.

## Current live extension inventory

Live host path:

- `/opt/gb10/stable-diffusion/Extensions`

Current directories:

- `Config-Presets`
- `model-keyword`
- `multidiffusion-upscaler-for-automatic1111`
- `sd_delete_button`
- `sd-webui-cardmaster`
- `sd-webui-detail-daemon`
- `sd-webui-incantations`
- `sd-webui-model-converter`
- `sd-webui-prompt-all-in-one`
- `sd-webui-refiner`
- `sd-webui-state-manager`
- `ultimate-upscale-for-automatic1111`

Current disabled extension setting:

- `sd-webui-state-manager` is already disabled in `config.json`.

Current startup evidence:

- no extension traceback in recent startup logs
- `sd-webui-prompt-all-in-one` starts a background API service
- only current startup warning is the accepted unauthenticated HF Hub notice

## Ownership policy

### First-class extensions

First-class means:

- source lives in this repo under `extensions/`
- provenance/license is documented
- `gb10/run.sh` syncs the repo-owned source into the host `Extensions/` mount before launch
- behavior changes are committed, reviewable, and validated with the image/runtime

Currently first-class:

- `sd-webui-incantations`
  - owns PAG, SEG, CFG-combiner, and Dynamic Thresholding / CFG-Fix behavior
  - replaces previous dependence on separate Incantations and Dynamic Thresholding checkouts

### External mounted extensions

External mounted extensions are tolerated only if they provide behavior we still need and are not yet worth adopting. Any external extension that materially affects generation quality, callback ordering, model loading, or high-value workflow behavior should either become first-class or be removed.

## Purge candidates now that A1111-Controller is canonical

These are primarily Gradio UI/QOL extensions and should be removed from the live A1111 runtime once any user data worth preserving is exported.

### `Config-Presets`

Recommendation: **purge**.

Reason:

- UI preset switching belongs in A1111-Controller or explicit API/client presets.
- It adds Gradio UI surface area without owning core generation behavior.

Preserve first if needed:

- `config-txt2img.json`
- `config-img2img.json`
- custom tracked component lists

### `model-keyword`

Recommendation: **purge after extracting useful mappings**.

Reason:

- automatic model/LoRA trigger-word insertion belongs in A1111-Controller's model/LoRA metadata layer.
- this extension is large relative to its runtime value and is UI-coupled.

Preserve first:

- `model-keyword.txt`
- `model-keyword-user.txt`
- `lora-keyword.txt`
- `custom-mappings.txt`
- `settings.txt`

Potential first-class replacement:

- not this extension as-is; instead, ingest useful mappings into the Controller metadata/catalog path.

### `sd_delete_button`

Recommendation: **purge**.

Reason:

- A delete button is UI-only and not needed for Controller-canonical operation.
- file/output management should live in Controller or ordinary filesystem tooling.

### `sd-webui-cardmaster`

Recommendation: **purge after confirming Controller covers the LoRA/card workflow**.

Reason:

- card browsing, LoRA activation text application, and extra-network interaction are exactly the kind of UI behavior A1111-Controller now owns for us.
- current config contains Card Master preferences, so confirm there is no unique workflow before removal.

Potential first-class replacement:

- do not adopt Card Master as an A1111 extension; move any still-useful LoRA metadata/activation behavior into A1111-Controller.

### `sd-webui-prompt-all-in-one`

Recommendation: **purge after exporting prompt favorites/history if wanted**.

Reason:

- prompt UI, tag grouping, prompt history, and translation helpers are Controller/UI concerns.
- it starts a background API service inside A1111, which is undesirable if Controller is the canonical frontend.

Preserve first if useful:

- `storage/favorite*.json`
- `storage/history*.json`
- `group_tags/*.yaml`
- translation config only if still intentionally used

Potential first-class replacement:

- do not adopt the extension wholesale; migrate useful prompt/tag data into Controller-owned prompt tools if needed.

### `sd-webui-state-manager`

Recommendation: **purge**.

Reason:

- already disabled.
- state/history restoration belongs in Controller.

Preserve first if useful:

- `history.txt`

## Keep or adopt decision candidates

These are not obviously superseded by A1111-Controller because they alter generation behavior or provide runtime utilities rather than just Gradio UI convenience.

### `sd-webui-detail-daemon`

Recommendation: **decide based on actual use; adopt if kept**.

Reason:

- it modifies generation behavior through sampling/noise scheduling.
- if Schwi actually uses it, it should be first-class because it affects output quality and interacts with callback/hot-path modernization.
- if not used, purge it rather than carrying another generation modifier.

Current evidence:

- no recent output metadata hits were found for `Detail Daemon` in the quick scan.

### `sd-webui-refiner`

Recommendation: **decide based on actual use; adopt if kept**.

Reason:

- it changes generation by swapping UNet/refiner behavior during sampling.
- A1111-Controller can expose controls, but the generation implementation still lives inside A1111 unless replaced by source work.

If kept:

- adopt as first-class or replace with repo-owned refiner handling in core/source modernization.

### `multidiffusion-upscaler-for-automatic1111`

Recommendation: **keep temporarily only if tiled generation/upscaling is still used; adopt or replace if kept long-term**.

Reason:

- it provides tiled diffusion and Tiled VAE behavior that may still matter for large image workflows.
- it already needed a GB10-local xformers/SageAttention/SDPA compatibility patch.
- patched generation/runtime behavior should not remain as an untracked external checkout.

Current repo note:

- `patches/mounted-extensions/multidiffusion-upscaler-for-automatic1111/0001-modern-attention-fallbacks.patch` records the local attention fallback patch.

If kept:

- adopt first-class or replace with a repo-owned tiled upscale path.

### `ultimate-upscale-for-automatic1111`

Recommendation: **keep temporarily if used; otherwise purge**.

Reason:

- it provides a known tiled upscale workflow, but overlaps with multidiffusion/tiled upscaling and possible Controller-driven workflows.
- if retained as a standard workflow, adopt or replace with a first-class implementation.

### `sd-webui-model-converter`

Recommendation: **remove from normal runtime; keep as an offline utility only if still useful**.

Reason:

- model conversion is not a normal generation-path extension and should not be loaded in the always-on A1111 runtime.
- if needed, run conversion as a deliberate maintenance/offline action, not as a mounted WebUI extension.

## Proposed purge/adoption plan

1. Export data from UI-only extensions that might contain useful user state:
   - Config Presets JSON files
   - model-keyword mappings
   - prompt-all-in-one favorites/history/group tags
   - state-manager history if desired
2. Move UI-only extension directories to a host-side quarantine outside `Extensions/` instead of deleting immediately.
3. Restart A1111 and smoke test API/model load.
4. Use A1111-Controller for normal prompt/model/LoRA/preset/state flows.
5. For remaining generation-affecting extensions, choose one of:
   - adopt first-class into `extensions/` with provenance and tests
   - replace with source-level A1111 modernization
   - purge if not actually used

Recommended immediate quarantine set:

- `Config-Presets`
- `model-keyword`
- `sd_delete_button`
- `sd-webui-cardmaster`
- `sd-webui-prompt-all-in-one`
- `sd-webui-state-manager`
- `sd-webui-model-converter`

Recommended hold pending use decision:

- `sd-webui-detail-daemon`
- `sd-webui-refiner`
- `multidiffusion-upscaler-for-automatic1111`
- `ultimate-upscale-for-automatic1111`

Already first-class / keep:

- `sd-webui-incantations`

## Cleanup boundary

Do not delete or quarantine live extension directories without a deliberate purge step. Extension removal is reversible if quarantined, but it changes the user-facing A1111 runtime and should be done as its own validated restart/smoke pass.
