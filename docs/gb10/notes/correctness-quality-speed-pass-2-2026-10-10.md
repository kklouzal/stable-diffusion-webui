# Correctness, quality and speed pass 2 — 2026-10-10

## Scope

A second end-to-end pass, from deploy12 (`b3c13fe2`, the same app code as deploy11). It excludes everything the
[2026-10-09 pass](correctness-quality-speed-pass-2026-10-09.md) and earlier notes record.

**Method.** Six read-only audits, each verified against the source:
- H: the GPU hot path, against the nsys profile
- N: end-to-end numerics of the production request against sgm, k-diffusion and ComfyUI
- R: request lifecycle and state across requests
- X: the extensions in the production request, including the never-audited Detail Daemon
- C: ControlNet, second pass
- S: cold start and first-request latency

Five implementation streams followed, each in its own worktree, then a GPU A/B session and a review round.

**Workload.** The operator's latest request (2026-10-10 00:52):
- img2img at 1280², "Multi: oi2", Align Your Steps 15
- ControlNet depth_zoe xinsir 0.66
- PAG 4 with SANF, SEG blur 6, Dynamic Thresholding mimic 4.5
- Detail Daemon, TeaCache, NGMS 5
- 8 LoRAs

Plain txt2img is 1024², DPM++ 2M + AYS 15, with the same LoRAs.

## GPU A/B (production stopped, port 7861, 1 s temperature monitor)

| Build | n-w1 | plain | Pixels |
|---|---|---|---|
| deploy12 | 11.357 s ± 0.073 | 4.288 s ± 0.011 | — |
| this pass, layout switches off | 11.911 s ± 0.114 | 4.284 s | changed by the fixes |
| **+ GroupNorm fast transpose + layout folds (shipped)** | **10.872 s ± 0.088** | **3.683 s ± 0.014** | **identical to the row above** |
| + cuBLASLt preference | 10.834 s | 3.671 s | changed (rejected) |
| + cuDNN heuristic mode B | 10.801 s ± 0.042 | 3.675 s | identical; within noise (rejected) |
| + `cudnn.benchmark` (two processes) | 10.874 / 10.859 s | 3.634 / 3.650 s | differ between processes |
| without ControlNet fp16 (shipped) | 10.410 s | 3.657 s | baseline for C-1 |

The fp16 ControlNet build (C-1) cost 0.46 s per request, and the owner had it reverted; see below.

**Sustained load.** 20 back-to-back n-w1 requests on the shipped configuration had a median of 11.05 s and a GPU peak
of 84 °C, with no errors and no host issue.

**Peak memory.** 15.9 GB allocated with the switches, 16.3 GB without. With `cudnn.benchmark` it was 28-46 GB.

## Speed

**GroupNorm fast transpose and layout folds**, on by default. Turn them off with `OPENCLAW_GN_FAST_TRANSPOSE=0` or
`OPENCLAW_LAYOUT_FOLDS=0`; an invalid value fails startup.

- **Why.** ATen's CUDA group_norm copies a channels_last input to NCHW internally, using a generic strided copy. At the
  VAE's full-resolution sizes that copy ran at about 14 GB/s.
- **The transpose.** A Triton tiled NHWC→NCHW transpose (128x128 tiles, 8 warps) feeds the unchanged ATen group_norm.
  It reaches about 215 GB/s, so a 1280² VAE GroupNorm takes 13 ms instead of 67 ms.
- **The folds.**
  - SiLU outputs that feed a conv are written channels_last.
  - The ResBlock `h + emb_out` is written NCHW for the GroupNorm that follows.
- **Effect.** The layout policy and every kernel's arithmetic are unchanged. GroupNorm on channels_last input and on
  its NCHW copy are bit-identical on GB10, which was checked in bf16, fp16 and fp32 at the production shapes. A VAE
  ResnetBlock at 1280² takes 84 ms instead of 202 ms.
- This is not the parked NHWC GroupNorm or all-NCHW work.

**Startup warm-up.** `OPENCLAW_WARMUP=generation-last` is run.sh's default.

- After a start, a background thread holding `queue_lock` replays the last generation once:
  - no images are saved;
  - the settings are restored;
  - no generation-last or snapshot write happens.
- Its status is at `GET /sdapi/v1/openclaw/warmup`.
- **Measured.** The warm-up finished 88 s after the container started. The first real n-w1 then took 19.7 s instead of
  about 81 s. The replayed request was a txt2img, so ControlNet and zoe still loaded on first use; after an img2img the
  warm-up covers those too.
- **Smoke test.** run.sh's smoke test now waits for the warm-up before the checks that need `queue_lock`.

**Smaller items:**
- LoRA file signatures are hashed concurrently: 0.74 s becomes 0.23 s on the first request.
- depth_zoe and Depth Anything v1/v2 build their parameters on the meta device, with no random init, and load with
  mmap. Their state is bitwise identical. A cold zoe load takes 0.9 s instead of 3.8 s.

## Correctness

**Request lifecycle (R stream):**
- A request's Denoise Ramp delta no longer becomes the default for later requests. The controller omits the ramp when
  its delta is 0, so before this fix a ramp could not be turned off.
- `override_settings` apply all-or-nothing, and a failed checkpoint or VAE override is reverted. Before, with
  `override_settings_restore_afterwards: false`, every later request answered 500.
- `override_settings` with an unknown checkpoint or VAE answer 422 instead of silently rendering on another model.
- Overridden options with an onchange callback (cross-attention optimization, fp8/MXFP8/NVFP4, `cache_fp16_weight`,
  ...) now take effect and are restored afterwards.
- img2img init cache: Tiled VAE requests bypass it, and its key covers the VAE's attention, SDPA backend, upcast and
  NHWC state.
- The refiner stage keeps the request's LoRAs.
- Hypertile follows a runtime attention-optimizer change.
- The SDPA-backend route no longer re-installs SDP over another optimizer.
- A skipped generation is not saved as the last generation.
- A request keeps the multi-sampler chain it first resolved.
- `GET /options` is race-free.

**Extensions:**
- **PAG SANF with Dynamic Thresholding.** SANF discards the Dynamic Thresholding result, and the infotext now says so:
  `PAG SANF note: Dynamic thresholding overridden by PAG SANF`. Composing them changed the image visibly (mean
  16.7 codes); the owner kept the established look.
- **TeaCache:**
  - lanes are keyed by (signature, ordinal), so NGMS's alternating batches can hit;
  - the step window uses sampler steps;
  - the infotext names its rescale fit (NoobAI-XL v-pred).
  - TeaCache does not engage while ControlNet is active.
- **clear-cond-cache.** The VAE torch.compile slot is removed.
  - It compiled only `forward`, while the core calls `decode`/`encode`, so it never ran.
  - Its `_orig_mod.*` keys made in-place checkpoint switches skip every VAE weight.
  - The route still answers, now with `vae: false` and the reason.
- **Detail Daemon** (deploy patcher `gb10/patch-detail-daemon.py`). A setting that would drive the model's sigma to 0
  or below now fails the request.
- **ControlNet:**
  - reference and ReVision noise comes from a per-request seeded generator;
  - the shuffle preprocessor's NumPy seed is the infotext seed.

## Measured, not adopted

| Item | Result |
|---|---|
| ControlNet in fp16 (C-1) | Residual error vs fp32 was 10x lower on the CPU measurement, but the image moved by a mean of 4.1 codes with no visible gain, and requests took 0.46 s longer. Reverted by owner decision. |
| Dynamic Thresholding composed under PAG SANF | Visible change. Reverted by owner decision. |
| cuBLASLt preference | Changes pixels, no gain. Reverted. |
| cuDNN heuristic mode B | Pixel-identical, gain within noise. |
| `cudnn.benchmark` | No warm gain, about 19 s more on the first request, 28-46 GB peak memory, nondeterministic across processes. The controller had turned it on; the owner turned it off (controller setting). |
| GEGLU dual GEMM, residual-add GEMM epilogue, regional torch.compile | Rounding-level, need custom kernels, estimates at most about 0.3-0.7 s; not pursued after the exact layout wins. |
| PAG replay at `middle_block[1]`, ControlNet NHWC zero conv, folded ControlNet weight scale | Below the measurement resolution. |
| ControlNet cond/uncond deduplication | Impossible: the context and `y` differ per row. |
| Lazy `pytorch_lightning` | The sgm/ldm model classes subclass it. |
| Infotext "Schedule type" for chains; AYS end point; timestep-only quantization | Upstream semantics; replay is exact. |

## Settings findings for the operator (no code change)

- **NGMS.** `s_min_uncond` 5 is above the img2img start sigma (1.84). The uncond pass is therefore skipped on every odd
  step from step 1, so CFG, SEG and Dynamic Thresholding are off for half the steps. This is upstream NGMS semantics.
- **SEG blur.** The SEG slider is an exponent: σ = 2^n query pixels. 6 is practically an infinite blur on the 40x40
  middle-block grid at 1280², so values above about 3 do the same thing.
- **Token merging.** `token_merging_ratio` 0.05 applies to img2img when `token_merging_ratio_img2img` is 0. That is
  A1111's documented semantics.
- **ControlNet detected map.** It is attached to every response (about 32 ms and 0.74 MB). The controller only reads
  the generated image; `control_net_no_detectmap=true` would drop the map.
- **VAE compile toggle.** The controller's `compile_vae` toggle now re-sends a no-op request each generation.
