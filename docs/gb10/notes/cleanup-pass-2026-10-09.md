# Cleanup pass — 2026-10-09

## Scope

The user asked for a pass over the entire source to remove dead code, duplication and stale surfaces, so that future
work starts from a clean codebase. Any bug found along the way was fixed in the same pass.

- **Base.** `c98dc893`, which equalled `origin/latest`: the deploy9 source (`75a94f59`) plus docs.
- **Result.** Branch `cleanup-integration` at `881a93f3`, followed by two follow-up streams:
  - S5a merges `tests/` into `test/` and moves the live-server tests to `test/live/`, which are skipped without
    `--base-url`.
  - S5b updates the docs and removes upstream leftovers.
- **Not deployed yet** (see "Deploy-time actions").
- **Evidence.** Everything is in `~/audit-artifacts/gb10-a1111-cleanup-20261009/`:
  - `LEDGER.md` holds decisions and status.
  - `area-*.md` are the audit reports. Finding ids such as 4-05 refer to them.
  - `S1/` to `S4/` and `S5b/` hold the stream logs.
  - `gate-*` and `probe-*` hold the gate results.

## Method

1. **Mechanical seeds:**
   - ruff (F, ERA001, PIE, B018, RUF100): 529 hits
   - vulture at 60% confidence: 440 candidates
   - pylint duplicate-code: 30 clusters

   Each candidate was verified by hand. Dynamic loading makes raw hits unreliable.
2. **Six read-only area audits:**
   - ControlNet runtime (area 1)
   - ControlNet assets (area 2)
   - GB10 core (area 3)
   - upstream core (area 4)
   - owned extensions and tooling (area 5)
   - tests, docs and root files (area 6)

   A removal had to clear every consumer:
   - the fork, including tests and non-Python files
   - dynamic use: callbacks, routes, option keys, getattr/importlib strings and path-loaded scripts
   - the host-installed third-party extensions
   - `~/a1111-controller` and `~/System-Statistics`
   - the companion repositories baked into the image
3. **Five implementation streams,** each in its own worktree from `c98dc893`, then merged in order:
   - S1 ControlNet
   - S2 GB10 core
   - S3a owned extensions
   - S3b tooling
   - S4 upstream core

   Post-merge fixes and a second-order ruff/vulture sweep ran on the merged tree.
4. **Integration gates** on every merge state. Both use the deploy9 image on the CPU only:
   - **CPU test gate** (`gate.sh`). It exports the tree with `git archive` and runs `tests test` plus each
     extension's tests in their own pytest process.
     - Baseline at `c98dc893`: 34 failed (all live-server tests without a server), 1238 passed.
     - At `881a93f3` (int3): the same 34 failed, 1430 passed, and every extension suite passed:
       clear-cond-cache 15, denoise-ramp 3, multi-sampler 16, ControlNet 73, Incantations 56, model-converter 22,
       TeaCache 37.
   - **CPU-only API probe** (`probe-cpu.sh`). It starts a `--network none` server with `--skip-load-model-at-start`,
     using the tree overlaid on the image and copies of the host third-party extensions. It records the script
     argument layouts, defaults, options, callbacks, X/Y/Z axes and routes, plus a 17-endpoint API snapshot.
     - Two runs at the base were byte-identical.
     - Every merge state was diffed against the base. Only the approved deltas below appeared.

## Owner decisions

- **ControlNet preprocessors.**
  - Install Depth Anything v1/v2, which works with the installed xinsir depth SDXL model.
  - Drop the preprocessors that could not run in the image: the 10 with missing dependencies, plus `ip-adapter_pulid`
    and its `facexlib` helper.
- **Features removed:**
  - the MXFP8 diagnostics probe and routes
  - ControlNet movie2movie, server-directory batch input, the AnimateDiff/SparseCtrl/PuLID paths and the offline
    converter CLIs
  - the `openclaw-conditioning-probe` extension
  - PAG's CFG Scheduler ("CFG Interval")
- **Legacy removed:**
  - Alt-Diffusion (4-71)
  - the never-run `prepare_environment` bootstrap (4-60). Its flags stay as no-ops, so `/cmd-flags` is unchanged.
  - the LDSR upscaler (4-68). Its option keys and `--ldsr-models-path` stay.
- **Kept on purpose:**
  - every checkpoint family a future model could need: SD1, SD2, SDXL, SD3, instruct-pix2pix
  - img2imgalt
  - the old emphasis
  - the alternative attention optimizers
  - the lowvram and IPEX code and flags
- **API surface: keep everything.** Every route, alias, request field, option key and CLI flag stays, except where a
  feature removal above takes it out.
- **Repo.**
  - Merge `tests/` into `test/`.
  - Drop the upstream leftovers: CHANGELOG, CITATION, CODEOWNERS, `.github`, and the Gradio README, which is rewritten.
  - Drop the stale docs: VISION, the April-May requirement docs, the pre-implementation NVFP4 notes and
    `.openclaw-audit`.
  - Condense STATUS.md.
  - Delete the untracked host clutter in the checkout.
  - Keep the ControlNet model weights in the checkout (run.sh mirrors them) and the local tag
    `quality-locked-baseline-20260812T004453Z`, which supplies the production infotext `Version` through
    `git describe`.
- **Upstream-core bugs.** Fix 4-01 to 4-08. For 4-05, take option (a): the `on_before_ui` X/Y/Z axes become selectable.

## Bugs fixed

ControlNet:
- `2b27a0d6`:
  - A legacy `control_net_*` request no longer changes the shared default units. Before this, a later request
    without ControlNet fields ran with the previous request's image.
  - Units beyond `control_net_unit_count` are no longer dropped.
  - Batch and AnimateDiff inputs fail validation, and image strings are never opened as server paths.
- `5152cdf3`: SparseCtrl checkpoints raise, and PuLID and FaceID Plus IP-Adapter models fail closed instead of failing
  later.
- `19afbd4a`:
  - `recolor_luminance` converts RGB, not BGR. **Output changes.**
  - `scribble_xdog` no longer wraps edges stronger than 127. **Output changes.**
  - revision_clipvision, revision_ignore_prompt and pidinet_scribble unload their models.
  - `ip-adapter-auto` with a missing preset raises the intended error.
  - The 12 unrunnable preprocessors now fail validation instead of failing at run time.
- `88e9a872`: with two or more recolor or inpaint_only units, each unit's post-processor uses its own map, not the last
  unit's.

GB10 core:
- `f0e8ff0b`, `a851cf35`:
  - VAE decode-graph identity probes fail closed, so a graph captured for another VAE cannot replay.
  - CUDA-graph status reads a copy, and enable/clear take the runtime lock.
  - In cache telemetry, the E11 sizes add up, and unknown reasons raise.
- `a08e0382`: the k-diffusion sigma cache key includes the quantize flag. sgm_uniform, normal and beta served stale
  sigmas after `enable_quantization` was toggled.
- `8f74814d`: the img2img init cache key carries the checkpoint and VAE lifecycle epochs. Under `--no-hashing`, an
  in-place replacement reused old init latents.
- `c1a40611`: the spandrel upscaler cache keys on file identity.
- `535b8cfe`: textual-inversion publishing includes file identity.
- `881a93f3`: the image-embedding disk cache keys on the file revision, not only the mtime.
- `34062b71` (4-04): a failed per-face restoration fails the request instead of returning unrestored faces.
- `e0be60ca`: a failing extra-network reset fails the request instead of being swallowed.
- `52219253`: the sampler/scheduler resolver cache is bounded (LRU of 256).
- `ec14f069`: TorchAO artifact renames are directory-fsynced, and the cache quota covers the whole backend root.

Owned extensions:
- `b62c514b`:
  - Dynamic Thresholding rejects every timestep sampler, including DDIM CFG++, which used to give silently wrong math.
  - The DynThres minimum sliders default to 0.0, so default-minimum requests no longer raise TypeError.
  - PAG raises when the model has no middle-block self-attention, instead of rendering without PAG.
- `4793cc56`:
  - The clear-cond-cache VAE compile slot compares objects, not `id()`.
  - A missing wrap target fails the install.
- `a63f37e1`: the converter's LoRA list matches `/sdapi/v1/loras`.

Upstream core:
- `32901290` (4-05): the `on_before_ui` X/Y/Z axes (SEG, PAG, DynThres, Hypertile) can be selected through the API.
- `1577fcbf`:
  - A failed TI/HN training run returns its error to `/train/*` (4-01).
  - HN no longer overwrites the target file with a partly trained net.
  - TI previews restore the torch RNG (4-02).
- `8befd36d` (4-03): ScuNET URL models are saved under their `.pth` name, so spandrel loads them.
- `68c045e5` and `029fbf96` (4-07): model downloads time out (connect 10 s, read 60 s) and close their response, so a
  stalled download cannot hold `queue_lock`.
- `c1feef15` (4-06): `--dump-sysinfo` and the launch log redact the credential flags. `28f00a02`: `--dump-sysinfo` no
  longer crashes without a readable settings file.
- `e103b67f`: a model merge that raises ends its job, so `/progress` is not left stuck.

Tooling:
- `73a65d5c`:
  - run.sh runs all its checks and rehearses the patchers before it removes the live container.
  - The rsync protects ControlNet `models/` (6-34). Deleting a model from the checkout used to delete it in
    production.
  - Tool caches are no longer deployed.
- `0e58a39a`:
  - `.dockerignore` is an allowlist, so host `config.json`, `params.txt` and `cache/` are no longer baked into the
    image (6-30).
  - Version floors compare as versions, so a pre-release cannot pass one.
  - The protected-names file is required.
  - `--snapshot` with `--out` is rejected.
- `4e4ae10f`: an empty patch set fails the build.
- `59ddf59a`: the MultiDiffusion patcher accepts pristine upstream `22798f6`, so a fresh install no longer aborts the
  deploy.
- `68a4e888` (6-02): pytest has no default base URL. The live-server tests used to target production :7860.

## Removals by area

`git diff --shortstat c98dc893 881a93f3`, grouped:

| Area | Files | Lines +/- | Main removals |
|---|---|---|---|
| ControlNet (`extensions/sd-webui-controlnet`) | 809 | +716 / -122,429 | annotator trees of the dropped preprocessors and other dead annotator code (-110,950); browser UI, movie2movie, batch/AnimateDiff/SparseCtrl/PuLID, installer, converter CLIs, upstream CI, web tests, examples, samples |
| other owned extensions | 29 | +1,561 / -1,872 | `openclaw-conditioning-probe` (-203); PAG CFG Scheduler; dead fallbacks. Tests moved in from `tests/` |
| `modules/` | 101 | +1,376 / -4,818 | MXFP8 diagnostics, Alt-Diffusion/XLM-R, macOS/MPS, Ascend NPU, ngrok, `uv_hook`, the `prepare_environment` bootstrap, `img2img.process_batch`, headless-UI stubs, write-only state |
| `extensions-builtin/` | 10 | +44 / -2,345 | LDSR (-2,179; a 24-line stub keeps its options and flag) |
| `scripts/` | 4 | +11 / -141 | `mxfp8_diagnostics_api.py` |
| `test/`, `tests/` | 85 | +2,924 / -2,328 | duplicate and dead tests; new oracle and regression tests |
| build/deploy (`gb10/`, `docker/`, `patches/`, Dockerfile, .dockerignore) | 25 | +709 / -1,583 | `gb10/config.json`, two patchers folded into `gb10/patchlib.py`-based ones, `patches/mounted-extensions/`, dead build knobs |
| root and upstream leftovers | 23 | +6 / -550 | `.github/`, CODEOWNERS, lint configs, `requirements*.txt`, Alt-Diffusion configs, the LDSR license section |
| **Total** | **1,086** | **+7,347 / -136,066** | |

S5b then deleted these:

| File | Lines |
|---|---|
| `.openclaw-audit/` | 5,346 |
| `CHANGELOG.md` | 1,085 |
| `CITATION.cff` | 7 |
| `screenshot.png` | 420 KB |
| `docs/gb10/VISION.md` | 93 |
| `docs/gb10/baseline_requirements.md` | 205 |
| `docs/gb10/enhanced_requirements.md` | 123 |
| `docs/gb10/notes/nvfp4*.md` | 458 |

S5b also rewrote README.md, STATUS.md, docs/gb10/README.md, launch/README.md and EXTENSIONS.md. `launch_utils.git_tag()`
no longer reads CHANGELOG.md. Its order is now: `A1111_VERSION_TAG`, then `git describe --tags`, then `<none>`.

## Observable API deltas

These are the int3 probe results at `881a93f3` against `c98dc893`. Nothing else changed: the remaining options keep
their values and order, `/cmd-flags` and the schemas are unchanged, and every kept infotext field keeps its index.

- **Routes gone (4):**
  - `GET /sdapi/v1/mxfp8-diagnostics` and `POST /sdapi/v1/mxfp8-diagnostics/run`
  - `GET /sdapi/v1/openclaw/conditioning-probe/self-test` and `GET /sdapi/v1/openclaw/conditioning-probe/snapshot`

  Their `app_started` callbacks went with them.
- **Scripts.** The `controlnet m2m` script is gone from `/sdapi/v1/scripts` and `/sdapi/v1/script-info`.
- **ControlNet.**
  - `/controlnet/module_list` went from 72 to 60. Dropped: `segmentation`, `oneformer_ade20k`, `oneformer_coco`,
    `normal_dsine`, `mediapipe_face`, `depth_hand_refiner`, `instant_id_face_embedding`, `instant_id_face_keypoints`,
    `ip-adapter_face_id`, `ip-adapter_face_id_plus`, `ip-adapter_pulid`, `facexlib`.
  - `/controlnet/control_types` lost `Instant-ID`. Segmentation now defaults to `seg_anime_face`, and the All, Depth,
    IP-Adapter and NormalMap lists are shorter.
  - The 33 ControlNet paste fields per tab are gone. Every one of them had index None.
- **Dynamic Thresholding.** "Minimum value of the Mimic Scale Scheduler" and "Minimum value of the CFG Scale
  Scheduler" default to 0.0 instead of null.
- **Incantations.**
  - "CFG Schedule Type" has the single choice `Constant`; it had 17.
  - The CFG-Interval inputs stay as placeholders, and `cfg_interval_enable=true` raises ValueError.
  - The four `[PAG] CFG` X/Y/Z axes are gone.
- **Upscalers.** LDSR is gone from `/sdapi/v1/upscalers` and from the 4 upscaler choice lists in script-info.
- **X/Y/Z.** The X/Y/Z type lists gained 29 axes (txt2img went from 53 to 82), appended after the existing ones: 4
  `[SEG]`, 5 `[PAG]`, 11 `[DynThres]` and 9 `[Hypertile]`.
- **Options.** The dynamic options `prioritized_callbacks_after_component`, `prioritized_callbacks_on_reload` and
  `prioritized_callbacks_script_unloaded` are no longer registered, because their only callbacks were dead ControlNet
  UI/batch hooks and `InputAccordion.reset`.
  - `GET /options` still echoes `config.json`.
  - A POST of those keys returns 422.
  - The controller posts only the keys it changes.
- **Startup.** A bad `OPENCLAW_CUDA_GRAPHS` value now fails startup.

## Deploy-time actions

1. **Rebuild the image** on an idle host, under the guarded build. In the new image, check:
   - the app tree matches the `.dockerignore` allowlist: no `test/`, `docs/`, `extensions/` or untracked host files
   - `depth_anything` and `depth_anything_v2` import
2. **Remove the retired host extension** before or with the deploy:
   `sudo rm -rf /opt/gb10/stable-diffusion/Extensions/openclaw-conditioning-probe`. run.sh never deletes a retired owned
   extension, so until then its two routes keep loading.
3. **Run the GPU live checks.** These need the user's explicit OK under the unified-memory guard. Use neutral prompts.
   1. The live API tests: `pytest test/live --base-url http://127.0.0.1:7860`.
   2. The eager-vs-graph fixed-seed comparison: `~/audit-artifacts/gb10-a1111-correctness-20261007/live/live_checks.py`.
   3. A fixed-seed txt2img comparison against deploy9 for the core and multi-sampler paths. `390ae826` deduplicated
      the initial-noise scaling, and no change is expected.
   4. A ControlNet `depth_anything` / `depth_anything_v2` request with the xinsir depth SDXL model.
   5. The output-changing ControlNet fixes: recolor_luminance, scribble_xdog, and two or more recolor or inpaint_only
      units.
   6. Optionally, an A/B of the cached autocast probe (`0a1207fc`). No speed gain is claimed.

The infotext `Version` keeps its production path: run.sh passes the checkout's `git describe --tags`.

## Deferred

These items need an owner decision or a follow-up:

- **3-32.** Orphan TorchAO sidecars, `.lock` files and stale leases are never reclaimed. Only the quota scope was
  fixed.
- **3-62.** `POST /sdapi/v1/openclaw/cuda-graphs` without `"enabled"` disables graphs, while the VAE route keeps its
  state.
- **Upstream-core items left alone by owner decision.** These include 4-08 (the unreachable unCLIP stats file) and
  4-64 (the extra-options-section headless snapshot); the full list is in `S4/LEDGER-S4.md`.
- **Follow-ups not done:**
  - a failed training run keeps partly trained weights in memory
  - `ddpm_edit` has dead VQModelInterface branches after the LDSR removal
  - `test_openclaw_cuda_graphs` leaks a grad-disabled state
- **Host leftovers.** Unmounted directories under `/opt/gb10/stable-diffusion` and the prompt-all-in-one quarantine
  copy (see STATUS.md).

## Kept on purpose

- **API keep-all surfaces:**
  - LDSR's option keys and flag (stub extension)
  - the bootstrap flags as no-ops
  - ControlNet's inert schema fields (batch fields, `pulid_mode`)
  - the Incantations CFG-Interval placeholders
- **The model families** listed under "Owner decisions".
- **The parked work:** the NHWC GroupNorm code and its runtime switch (off), and the `--opt-channelslast` default. See
  [performance-pass-2-2026-10-07.md](performance-pass-2-2026-10-07.md).
- **The cuda_graphs `_original_forward` bypass guard,** a generic convention for third-party hooks.
- **`html/licenses.html` and `LICENSE.txt`.**
