# Performance pass 2 — 2026-10-07

Scope: end-to-end generation speed after the correctness audit, with no humanly perceptible quality change. GPU work ran
only while the host was idle (operator rule) and with neutral prompts. Evidence, scripts and traces:
`~/audit-artifacts/gb10-a1111-perf2-20261007/` (LEDGER.md, findings/S1-S4, profiles/, runs/).

## What the profile showed

torch.profiler traces of the deployed build (CPU+CUDA): the operator's img2img workload (1280², "Multi: oi" + AYS 15
steps, ControlNet depth_zoe, PAG, SEG, Dynamic Thresholding, Detail Daemon, TeaCache) keeps the GPU busy 99.6% of the
request, a plain txt2img 99.8%. CPU launch overhead is hidden, so CUDA graphs for the extension stack were dropped as a
target. GPU time: GEMMs (cuBLASLt nvjet) 49%; memory-format copies 12% (20% on plain txt2img) because CUDA GroupNorm
only takes NCHW and `--opt-channelslast` makes every GroupNorm copy to NCHW and every conv copy back; the GEGLU
feed-forward's gelu and mul on strided halves ~12%; flash attention 9%.

## Adopted (on by default)

| Change | Class | Effect |
| --- | --- | --- |
| Fused Triton GEGLU (`modules/openclaw_fused_geglu*.py`, `OPENCLAW_FUSED_GEGLU`) | bitwise on device | 1.6x faster feed-forward activation; links the CUDA toolkit's libdevice so erff matches ATen (Triton's bundled libdevice has a different erff and moved 4/65536 bf16 values by 1 ULP) |
| Dynamic Thresholding: no kernels that cannot change a bit; SEG writes blurred queries in place; TeaCache keeps one fp32 residual copy | bitwise | 26 -> 17 DT ops/step; ~164 MB/step less SEG traffic |
| Parallel PNG writer for disk saves (`modules/png_writer.py`, option `png_parallel_encoder`) | pixel-exact, same chunks/filters/deflate level 6 | 1280² save 428 -> 56 ms, 1536² 544 -> 67 ms; file size +0.05% |
| Inline API images decoded once per request; cv2 RGB<->RGBA for ControlNet inputs | exact | ~56 ms + ~3 ms per 1280² img2img with a ControlNet image |
| override_settings restore covers options never saved to config.json | fix | an override (e.g. profiling) no longer outlives its request |

Measured on the GPU (fixed seed, warm, neutral prompts): the new build produces pixel-identical images to the previous
deploy on all four workloads, and is faster: plain txt2img 4.49 -> 4.18 s, txt2img full stack 9.19 -> 8.68 s,
img2img workload 14.52 -> 13.62 s, hires fix 17.90 -> 16.84 s (save_images=false in these runs; real requests that save
to disk also gain ~0.37 s at 1280²).

## Measured but not adopted

- **All-NCHW deploy** (no `--opt-channelslast`): 8.5-16% faster, but the host hard-locked twice during hires-fix
  requests on it (2 of 3 attempts). Ruled out.
- **NHWC GroupNorm(+SiLU) Triton kernels** (`modules/openclaw_nhwc_groupnorm*.py`, runtime switch
  `POST /sdapi/v1/openclaw/nhwc-groupnorm`, default off): deterministic, at least as accurate as torch, 3-17x faster per
  GroupNorm and 21-30% faster per request, but the host hard-locked on the first hires-fix request with it on. Stays off;
  enabling it needs a root cause for the hang first.
- apex.contrib.group_norm: rejected (fp32 atomicAdd two-pass statistics are nondeterministic run to run).
- cuDNN SDPA (8-16% slower than flash on GB10), CUDA graphs for SEG/PAG/ControlNet (GPU-bound), PNG level 1 (+10-13%
  files across the 4 MB export threshold), cudnn.benchmark (cross-process nondeterminism).
- Small exact GPU items left for a later GPU session: PAG perturbed attention computing only V (~0.5-1 ms/step),
  ControlNet skip concatenation into a preallocated buffer (~1.2 ms/call), PAG replay reusing decoder skip sums
  (~0.9 ms/step). TeaCache does not engage while ControlNet is active (its threshold is a no-op on ControlNet requests).

## Host stability

Three hard lockups on 2026-10-07 (15:43, 15:59, 17:31), each during a hires-fix request in a non-default kernel
configuration (NCHW twice, NHWC GroupNorm once), GPU at ~96% and 80-84 C, >90 GiB memory free; the hard watchdog is
disabled so nothing was logged. Default-kernel hires requests never hung. Experimental deploys now run with restart
policy "no", a fsynced 1 s monitor and a low-memory interrupt.
