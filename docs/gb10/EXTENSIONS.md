# GB10 extensions

The host directory `/opt/gb10/stable-diffusion/Extensions` is mounted as A1111's `extensions/` directory. The
extensions in it are of two kinds:

- **Owned**: the source lives in this repo under `extensions/`. On every deploy, `gb10/run.sh` mirrors each one into
  the host directory (see [launch/README.md](launch/README.md#deploy-gb10runsh)). The image does not contain them.
- **Third-party**: installed only on the host. run.sh patches them in place at deploy time.

A1111-Controller is the frontend. UI-only extensions do not belong here. Any extension that affects generation quality,
callback ordering or model loading should be owned source, not an opaque host checkout.

## Owned extensions

| Directory | Provenance | Provides |
|---|---|---|
| `openclaw-clear-cond-cache` | GB10-owned | Controller helper routes under `/sdapi/v1/openclaw/` (listed below) and the backend activity status fed by hooks on model, VAE and LoRA loading |
| `openclaw-denoise-ramp` | GB10-owned | the `OpenClaw Denoise Ramp` script, which wraps the k-diffusion sigma schedule for img2img |
| `openclaw-multi-sampler` | GB10-owned | the `OpenClaw Multi-Sampler` script and `/sdapi/v1/openclaw/multi-sampler` (GET), `.../custom` (POST), `.../custom/{name}` (DELETE) and `.../preview` (POST). Saved chains live in the extension's `data/`, which deploys preserve |
| `sd-webui-controlnet` | `Mikubill/sd-webui-controlnet` v1.1.455 (`56cec5b`), GPL-3.0 | API-only ControlNet; see its [README](../../extensions/sd-webui-controlnet/README.md) |
| `sd-webui-incantations` | Incantations (GPL-3.0) plus `mcmonkeyprojects/sd-dynamic-thresholding` (MIT) | PAG, SEG, CFG-combiner, and Dynamic Thresholding / CFG-Fix; see its [README](../../extensions/sd-webui-incantations/README.md) |
| `sd-webui-model-converter` | `Akegarasu/sd-webui-model-converter` at `a8c04410` | checkpoint and LoRA conversion: `/sdapi/v1/openclaw/model-converter/options` (GET) and `.../convert` (POST) |
| `sd-webui-teacache` | `feffy380/sd-webui-teacache` at `a8cecf28`, MIT | SDXL TeaCache acceleration, off by default |

The routes of `openclaw-clear-cond-cache` (see its [README](../../extensions/openclaw-clear-cond-cache/README.md)):

| Method | Path |
|---|---|
| POST | `/sdapi/v1/openclaw/clear-cond-cache` |
| GET | `/sdapi/v1/openclaw/cond-cache` |
| POST | `/sdapi/v1/openclaw/token-count` |
| POST | `/sdapi/v1/openclaw/token_counter` (alias of `token-count`) |
| GET, POST | `/sdapi/v1/openclaw/torch-compile` |
| GET | `/sdapi/v1/openclaw/backend-status` |
| GET, POST | `/sdapi/v1/openclaw/cudnn-benchmark` |
| POST | `/sdapi/v1/openclaw/model-merge` |
| GET | `/sdapi/v1/openclaw/training-templates` |

ControlNet notes:

- **Models.** Model weights live in the checkout's `extensions/sd-webui-controlnet/models/`. Git ignores them; the
  identity of each is pinned by a committed `.sha256` sidecar. See [gb10/controlnet-models.md](../../gb10/controlnet-models.md).
- **Preprocessor weights** download on first use into `annotator/downloads/`. Deploys preserve both directories on
  the host.
- **Supported preprocessors.** The 2026-10-09 cleanup dropped the preprocessors that could not run in this image.
  `module_list` went from 72 to 60.
- **Depth Anything.** The Depth Anything v1/v2 packages are installed in images built from 6bbe3a96 on.

Incantations notes:

- PAG's CFG Scheduler ("CFG Interval") was removed on 2026-10-09.
- Its four script inputs remain as placeholders, so positional `alwayson_scripts` arguments keep their indices.
  `cfg_interval_enable=true` raises an error.

Rules for owned code:

- Make changes here.
- Keep each upstream's license and provenance notes.
- Patch generation math and cache behavior conservatively: it changes images.

**Retired.** `openclaw-conditioning-probe` (cache-coherency diagnostics routes) was removed on 2026-10-09. run.sh
never deletes a retired owned extension, so remove the host copy
`/opt/gb10/stable-diffusion/Extensions/openclaw-conditioning-probe` by hand.

## Third-party extensions

A1111-Controller uses these, so they stay installed on the host:

| Directory | Deploy-time patch (`gb10/`, on the `patchlib.py` contract) |
|---|---|
| `multidiffusion-upscaler-for-automatic1111` | `patch-multidiffusion-performance.py` |
| `ultimate-upscale-for-automatic1111` | `patch-ultimate-upscale-state-lifecycle.py` and `patch-ultimate-upscale-subcanvas.py` |
| `sd-webui-detail-daemon` | none |

What the patches do:

- **`patch-multidiffusion-performance.py`.** Its original text is upstream `22798f6`, so a fresh install patches
  cleanly. It covers:
  - the MultiDiffusion terminal-tile fix: upstream's tile origins with only the last one pinned to the edge, so the
    last latent row or column is always covered; identical to upstream wherever upstream covers it
  - the Tiled VAE attention fallbacks (previously `patches/mounted-extensions/`)
  - the Tiled VAE and MultiDiffusion performance changes, including ControlNet control tiles built once per request
  - noise inversion: this batch's prompts with extra networks parsed out, SDXL size conditioning, and an
    inverted-noise cache that never outlives a request and is reused only for exact matches
  - region prompt control on SDXL/SD3 fails before any work instead of with a TypeError mid-sampling
- **`patch-ultimate-upscale-state-lifecycle.py`** fixes how Ultimate Upscale handles job state and reports results:
  - The request's job owns the shared state; there is no nested `state.begin()`/`end()`, so an interrupt sent while
    the upscaler runs stops the tiles.
  - `override_settings` apply once around all tiles, so an `sd_vae` override no longer reloads the VAE twice per tile.
  - A pass that runs no tile (interrupted, or a single tile row or column) keeps the infotext and does not repeat the
    image.
- **`patch-ultimate-upscale-subcanvas.py`** processes every tile at its crop region's own size:
  - The padding grows by 0-7 px to reach a multiple of 8, so tiles are no longer resampled down and back up and the
    band-pass seam is no longer stretched.
  - Redraw tiles are exactly `tile_width x tile_height`.
  - Each tile gets a window of the canvas, bitwise identical to processing the whole canvas at those sizes.
  - This changes images on purpose compared with upstream and deploy10.

The Ultimate Upscale patchers apply exact blocks to upstream Coyote-A master `2322caa`. Each patcher accepts the
original text or its already-patched text, and fails the deploy on anything else. A patcher whose blocks changed
since deploy10 also upgrades the deploy10 text in place (`patchlib` `previous`: the target is reverted to upstream,
proven by a round trip, and patched again); `--check` reports such a target as outdated. The `PREVIOUS`/`DEPLOY10`
blocks can be deleted once every host runs this release. run.sh rehearses every patcher on a scratch copy before it
stops production.

Candidates for adoption as owned source:

- MultiDiffusion, which is in the generation and runtime hot path and already patched
- detail-daemon, which changes sampling noise
- Ultimate Upscale

## History

- **2026-05.** These UI-only or Controller-superseded extensions were removed from the host:
  - `Config-Presets`
  - `model-keyword`
  - `sd_delete_button`
  - `sd-webui-cardmaster`
  - `sd-webui-prompt-all-in-one`
  - `sd-webui-state-manager`

  The Controller had used prompt-all-in-one's `/physton_prompt/token_counter`. It now calls
  `/sdapi/v1/openclaw/token-count`. One quarantine copy is still on the host, waiting for an owner decision:
  `/opt/gb10/stable-diffusion/Extensions.quarantine/20260503-194044/sd-webui-prompt-all-in-one`.
- **Earlier adoptions.** Dynamic Thresholding was folded into `sd-webui-incantations`. TeaCache, ControlNet and the
  model converter became owned source.
