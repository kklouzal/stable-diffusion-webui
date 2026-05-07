# MXFP8 img2img final baseline — 2026-05-06

This note records the final end-to-end GB10 A1111 img2img/MXFP8 baseline after the MXFP8 LoRA work and SEG/PAG stabilization pass.

## Scope

Validated workflow:

- SDXL img2img and txt2img through the A1111 API.
- GB10 Blackwell runtime with PyTorch/TorchAO/Triton MXFP8 support.
- Active LoRA stacks under the MXFP8 merge-then-quantize path.
- SDPA/SageAttention backend switching smoke coverage.
- SEG/PAG extension behavior with A1111 paired and unpaired CFG execution paths.

## Authoritative implementation alignment

The current implementation is aligned with the relevant upstream semantics:

- MXFP8 uses E4M3 FP8 values with one E8M0 power-of-two scale per 32-value block.
- TorchAO `MXDynamicActivationMXWeightConfig` defaults to 32-element microscaling blocks, BF16 activation/output integration, `KernelPreference.AUTO`, and `ScaleCalculationMode.RCEIL`.
- Blackwell block-scaled matmul paths require hardware-oriented scale layouts/swizzling; TorchAO/MSLK own that packing rather than repo-local code.
- PyTorch SDPA remains a BF16 attention kernel path; MXFP8 Linear quantization does not replace SDPA itself.
- Selective quantization is preferred over full-model quantization for diffusion quality/performance tradeoffs.

## Current runtime baseline

Validated container/runtime:

- image tag: `local/gb10-a1111:latest-mxfp8-dev`
- live container: `gb10-a1111-latest-mxfp8`
- A1111 endpoint: `http://127.0.0.1:7860`
- checkpoint: `test2.safetensors`
- VAE: `ftasticVAE_v10.safetensors`
- attention backend after validation: `sdpa`
- `batch_cond_uncond`: `true`
- `s_min_uncond`: `0`
- MXFP8 storage: `Enable for SDXL`
- MXFP8 LoRA behavior: merge active LoRA deltas into BF16 master weights, then quantize to MXFP8

Runtime package evidence from `/sdapi/v1/mxfp8-diagnostics/run`:

- GPU: `NVIDIA GB10`
- CUDA capability: `[12, 1]`
- Python: `3.12.3`
- PyTorch: `2.13.0`
- PyTorch CUDA: `13.2`
- TorchAO: `0.17.0`
- Triton: `3.7.0`
- `torch.float8_e4m3fn`: available
- `torch.float8_e8m0fnu`: available
- `torch._scaled_mm`: available
- `torch.nn.functional.scaled_mm`: available
- TorchAO `ScalingType.BlockWise1x32`: available
- TorchAO `SwizzleType.SWIZZLE_32_4_4`: available

## MXFP8 quantization policy

Current policy-allowed quantization is intentionally conservative:

- Quantize policy-eligible `torch.nn.Linear` layers outside attention and conditioner regions.
- Skip VAE, conditioner, self-attention, and cross-attention Linear layers.
- Keep SDPA/SageAttention attention computation in BF16.
- Use RCEIL scaling mode.
- Keep LoRA base BF16 masters for MXFP8-managed layers so active LoRA deltas can be merged from clean full-precision weights before requantization.

Latest A1111 integration audit:

- total Linear modules: `911`
- eligible/quantized Linear modules: `183`
- skipped Linear modules: `728`
- skipped reasons:
  - `self_attention`: `280`
  - `cross_attention`: `280`
  - `conditioner`: `168`
- active BF16 LoRA-managed MXFP8 layers after preparation: `0`

Shape guard evidence:

- `in_features` must be divisible by `32` for MXFP8 block scaling.
- `out_features` must be divisible by `16` for the active TorchAO/kernel path.
- `out_features` divisible by `16` but not `32` is valid.
- batch/M dimension not divisible by `16` is valid in the smoke matrix.
- dimensions divisible by `32` but not `128` are valid.

Scaling evidence:

- RCEIL smoke mean absolute error: `0.0165238119661808`.
- FLOOR smoke mean absolute error: `0.025865375995635986`.
- Keep RCEIL as the default.

## LoRA behavior

Default behavior: merge active LoRA deltas into BF16 master weights once, then quantize to MXFP8.

Rationale:

- Normal A1111 LoRA mutation expects ordinary mutable `torch.nn.Parameter` tensors.
- TorchAO `MXTensor` weights do not compose safely with the normal in-place LoRA path.
- The current path keeps BF16 base weight/bias masters, applies the active LoRA delta stack to BF16 once when the active LoRA set changes, then requantizes the touched layer back to MXFP8.
- The old selectable BF16 LoRA fallback was removed after merge-then-quantize became the only validated fast path.

## Attention backend posture

Default: `sdpa`.

Rationale:

- SDPA is stable and least surprising for the A1111/SDXL/img2img workflow.
- Runtime coverage confirms SDPA Q/K/V tensors remain BF16.
- SageAttention2++ and SageAttention3 Blackwell are available and passed smoke generation, but did not provide enough workflow-level improvement to replace SDPA as the default baseline.

## SEG/PAG behavior

The current SEG/PAG fixes are intentional and should not be simplified into generic tensor splitting.

Correct semantics:

- SEG mutates only the conditional half of a paired CFG attention batch.
- A1111 may run singleton/unpaired cond/uncond passes when token lengths mismatch, when batch-cond-uncond is disabled, for skip-uncond paths, or inside hidden extension passes.
- SEG must skip unpaired batches rather than attempting to split them.
- SEG should only enable during denoiser steps when `shared.opts.batch_cond_uncond` is true.
- PAG should suspend SEG during `pag_inner_model_x_out(...)` and restore the previous `seg_enable` state afterwards.

Do not replace the paired-batch guard with `torch.tensor_split`; that hides crashes while preserving wrong semantics.

## End-to-end validation matrix

A 9-case A1111 generation matrix completed successfully with the same recovered anime prompt, negative prompt, and 13-LoRA stack/strengths:

1. `01-txt2img-sdpa-dpmpp2msde-auto` — `119.03s`
2. `02-txt2img-sdpa-eulera-karras` — `74.88s`
3. `03-txt2img-sdpa-dpmpp3msde-exponential` — `75.03s`
4. `04-txt2img-sage2-dpmpp2msde-karras` — `75.16s`
5. `05-txt2img-sage3-dpmpp2msde-karras` — `76.05s`
6. `06-txt2img-sdpa-seg-pag` — `148.95s`
7. `07-img2img-sdpa-baseline` — `38.80s`
8. `08-img2img-sage3-eulera-karras` — `38.88s`
9. `09-img2img-sdpa-seg-pag` — `68.42s`

Artifacts were written under `/tmp/a1111-thorough-tests` in the GB10 runtime and copied to the Orion workspace under `tmp/a1111-thorough-tests/` during validation.

## Current recommendation

No material runtime correction remains for the current img2img baseline.

Keep as the default baseline:

- `sdpa` attention backend
- `Merge LoRA then quantize to MXFP8`
- selective MXFP8 Linear quantization outside attention/conditioner regions
- RCEIL MXFP8 scaling
- SEG paired-batch guard plus PAG hidden-pass SEG suspension

Future changes should treat this note as the known-good baseline and should run an equivalent A1111 API matrix before changing image-affecting math, quantization coverage, attention backend defaults, or guidance-extension semantics.
