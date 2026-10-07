# Static performance pass — 2026-10-06

Scope: every SDXL txt2img/img2img code path in the `local/gb10-a1111:latest` source (core, owned extensions,
host-mounted extensions via deploy patchers, build/deploy), assuming all extensions may be enabled.
Contract: no humanly perceptible output change. Items are labelled **exact** (bitwise in deterministic mode),
**equivalent** (numerically equivalent: different valid kernel/batching, at most ULP-level differences), or
**fix** (behaviour change that corrects a defect). Nothing that approximates sampling math was made a default.

Full audit evidence (findings per area, ledger, test logs, GPU microbenchmarks):
`~/audit-artifacts/gb10-a1111-perf-static-20261006/` (LEDGER.md, findings/, t-*.txt, runs/).

## Correctness defects fixed

- **Multi-sampler NaN** (fix): a later chain stage overwrote only its first sigma with the handoff sigma, so the
  schedule could rise (e.g. "Multi: oi" img2img 0.1011 -> 0.1626) and DPM++ 2M SDE produced an all-NaN latent after
  the full run. Non-rising splices stay bit-identical; rising ones are rebased in log-sigma; stage validation rejects
  rising or non-finite sigmas before any UNet call.
- **ControlNet hook leak** (fix): the UNet hook was removed only on success, so a failed generation left a hook with
  the old hint installed; later requests crashed (shape mismatch) or silently applied the stale map, with CUDA graphs
  and TeaCache disabled. `restore_leaked` heals the UNet at the next ControlNet entry and forward_webui is inert outside
  sampling.
- ControlNet `mark_prompt_context` no longer mutates cached conditioning (stale ±1024-token marks; cond cache reset
  every forward). The folded deploy cache patch's `cacheable` field shadowed subclass flags, so the preprocessor cache
  never hit; fixed.
- Strict API sampler names (unknown names returned DPM++ 2M silently); upscaler load failures raise instead of a
  silent LANCZOS fallback; hires cond-cache key includes hires size and refiner aesthetic scores; "TI hashes"/"Emphasis"
  infotext restored on cond-cache hits; soft inpainting full-res uncrop and per-request mask reset.
- CUDA graphs: bypass per-request monkeypatches (MultiDiffusion/MoD/DemoFusion/Tiled VAE), UNet hypertile and tomesd;
  the cache entry owns the sampler wrapper and schedule tensors the graph reads; VAE graphs bypass Tiled VAE.
- Races: the MXFP8 startup probe runs under `queue_lock` with local RNG and no process-wide SDPA monkeypatch;
  model-converter and conditioning-probe endpoints take the lock; blocking clear-cond-cache work leaves the event loop.

## Per-step (UNet/guidance) changes

| Change | Class | Basis |
| --- | --- | --- |
| PAG hidden pass runs only the cond rows the combiner reads | equivalent (~1 ULP where rows are cut) | static: P1, ~-0.42 s/step on the captured workload |
| PAG pass reuses the main call's ControlNet residuals/marks | exact at equal batch | static: ~-0.13 s/step |
| PAG pass resumes at `middle_block` from recorded encoder rows (off with hypertile U-Net) | exact at equal batch | static: ~-0.09 s/step |
| SEG blur as two fp32 GEMMs instead of a 41x41 depthwise conv | equivalent (closer to exact blur) | static: 69 -> ~3 GFLOP/step |
| UNet GroupNorm/LayerNorm run natively in bf16 under autocast (`UNET_BF16_NATIVE_NORMS`) | LN bitwise; GN equivalent (eps rounds to bf16) | GPU: GN 2.2-4.7x, LN 6.7-12.7x per op; error at the bf16 floor vs float64 |
| ControlNet: guided hint once per hint, placement once per hook, device timestep freqs, unet dtype, first residual assigned, contiguous context once, cond-rows-only for "more important"/GAP | exact (CN11 equivalent) | static |
| SDPA backend string parsed once; `sdpa_kernel` skipped for the default set | exact | static |
| sgm SpatialTransformer output copy removed | exact | CPU differential |
| LoRA: wanted names built once per publish; no-LoRA early-out; graph keys use `source_key` instead of hashing files per call | exact | static |
| CUDA graphs: one warm-up per new key, shared pool, LRU eviction | exact | GPU unit tests |

## Per-request changes

- VAE mid-block attention uses 4-D q/k/v so a fused kernel replaces the math path (5 GB fp32 score matrix at 1280²):
  equivalent, GPU-tested with bounded peak memory. Same for Tiled VAE tiles (deploy patcher).
- No mid-request `torch_gc()`; one release at job end. generation_last snapshot encodes each image once and keeps
  inline PNGs; API/snapshot PNGs at zlib level 1 (lossless; disk saves unchanged). Init-image cache keys on uint8.
- Hires: skip the full-size decode a plain model discards; decode on device. Upscalers: LRU-2 model cache,
  uint8-only device transfers. Mask blurs in parallel. ControlNet preprocessor result cache for deterministic
  preprocessors; no gc/torch_gc in ControlNet postprocess; uint8 hint upload.
- Startup/process: `gc.freeze()` before model load (a full collection drops from ~170 ms to ~0; several extensions
  collect per request); torch threads match the cpuset; no gzip for loopback clients (~165 ms per 7 MB response);
  precompiled bytecode; compile-cache namespace keyed by torch/Triton/CUDA/driver so app-only deploys keep warm JIT
  caches; checkpoint switch loads in place on unified memory (no 7 GB CPU round trip) when the cache limit is 1;
  ControlNet models built on the meta device (no random init).
- Tiled VAE/MultiDiffusion/USDU (deploy patchers, fail closed): no host round trips or gc in Tiled VAE, precomputed
  MoD weights, per-tile sub-canvas windows in USDU (bitwise over 70 layouts). Soft inpainting histogram filter
  vectorized (bit-identical, ~65x).
- Build: the wheelbuilder no longer copies the app source, so app-only builds reuse the dependency closure instead
  of re-resolving released packages; `.dockerignore` excludes nested bytecode and agent worktrees.

## Deferred (need GPU A/B evidence) and rejected

Deferred: NHWC GroupNorm+SiLU fusion, regional torch.compile, cuDNN heuristic mode B, channels_last A/B for
UNet/VAE/ControlNet, folding PAG rows into the main call, graphs for SEG/PAG/masked denoisers, expandable_segments,
tcmalloc, startup warm-up, masked init cache.
Rejected: cross-attention K/V caching (context rebuilt per step; PAG's `to_v` hook), fused QKV (breaks hooks),
global bf16 conditioning (fp32 consumers change), dropping autocast, `--disable-nan-check` (breaks the VAE precision
fallback), `cudnn.benchmark` (nondeterministic), ControlNet flip batching (not exact), sampler micro-optimizations
hidden at ~95% GPU utilization.

## Open quality decisions (not changed)

SDXL size/aesthetic embeddings and UNet timesteps are built in bf16 (1500 -> 1504; t >= 256 quantized); fixing either
changes outputs. The API echoes ControlNet input images in `parameters` (contract change to remove).
`configs/sd_xl_inpaint.yaml`/`sd_xl_v.yaml` reference a class absent from the baked sgm.

## Verification status (at push)

- CPU suites in the image (`tests test` and every `extensions/*/tests`, ControlNet web_api excluded): no new failures
  against the pre-change baseline; 8 previously failing extension tests now pass. The only new core entries are four
  LoRA identity tests that error from pre-existing `tests/test_teacache_extension.py` `sys.modules` stub pollution, like
  every other test in that file; the file passes 38/38 on its own.
- GPU unit tests (before the merge of the PAG workstream): UNet/attention/graph suites 127/127; full core and
  extension suites with `--gpus all` showed no new failures versus the same commit range on the pre-change tree.
  bf16 native norms were measured on GB10 (`test/benchmark_unet_norms.py`).
- Ruff on the 105 changed Python files: 82 findings vs 83 on the same files before (no new rule categories).
- NOT yet run: any end-to-end generation on the new code, fixed-seed image comparison against the previous build, and
  per-workload s/step / peak-memory A/B. Two host lockups on 2026-10-06 coincided with GPU replay jobs on this host,
  so end-to-end validation is left to the operator. Rollback image: `local/gb10-a1111:pre-perf-20261006`.
