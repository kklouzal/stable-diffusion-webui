# NVFP4 / MXFP4 codepath analysis

Date: 2026-05-07
Branch: `latest-mxfp8-dev`
Scope: baseline for adding separate NVFP4 enablement controls alongside the stabilized MXFP8 path.

## Executive conclusion

NVFP4 can reuse the current MXFP8 integration shape, but it should not be implemented as a literal one-line toggle inside the existing MXFP8 code. The stable path is to factor the shared TorchAO-Linear orchestration concepts and keep format-specific backends separate.

Recommended first implementation:

- keep MXFP8 as the default/known-good path
- add `nvfp4_storage` and, optionally, `nvfp4_linear_coverage`
- make native FP8, MXFP8, and NVFP4 mutually exclusive
- reuse the current conservative Linear region policy and model reload discipline
- keep NVFP4 cache, diagnostics, and LoRA active-config state separate from MXFP8 at first
- do not implement mixed per-layer MXFP8/NVFP4 in the first pass

MXFP4 is a different story. TorchAO exposes MXFP4 through `MXDynamicActivationMXWeightConfig(... torch.float4_e2m1fn_x2 ...)`, but the live GB10 container rejected MXFP4 with `NotImplementedError: MXFP4 scaling only supported in CUDA for B200/B300`. On GB10, NVFP4 is the practical FP4 route to prototype first.

## Live TorchAO findings

The current GB10 container has:

- PyTorch `2.13.0.dev20260505+cu132`
- TorchAO `0.17.0`
- `torch.float4_e2m1fn_x2`, `torch.float8_e4m3fn`, and `torch.float8_e8m0fnu`
- `torchao.prototype.mx_formats.NVFP4DynamicActivationNVFP4WeightConfig`
- `torchao.prototype.mx_formats.NVFP4WeightOnlyConfig`
- `torchao.prototype.mx_formats.MXDynamicActivationMXWeightConfig`

Observed smoke behavior:

- MXFP8 `MXDynamicActivationMXWeightConfig(float8_e4m3fn)` worked on BF16 Linear inputs.
- NVFP4 `NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=True, use_triton_kernel=True)` worked on BF16 Linear inputs and produced finite BF16 output.
- NVFP4 with `use_triton_kernel=False` also worked on small Linear tests.
- MXFP4 `MXDynamicActivationMXWeightConfig(float4_e2m1fn_x2)` failed on GB10 with `MXFP4 scaling only supported in CUDA for B200/B300`.
- NVFP4/FP4 paths require BF16; previous direct tests showed FP16 unsupported.

## Codepaths that can be safely reused

### Option shape and reload wiring

Reusable:

- `modules/shared_options.py`
  - existing `mxfp8_storage` shape: `Disable` / `Enable for SDXL` / `Enable`
  - existing `mxfp8_linear_coverage` choices: `unet_other`, `self_attention`, `cross_attention`, `conditioner`
- `modules/initialize_util.py`
  - onchange reload style using `reload_model_weights(forced_reload=True)` for TorchAO tensor-subclass modes

NVFP4 should add sibling options rather than replace MXFP8 options:

- `nvfp4_storage`
- `nvfp4_linear_coverage` or a later generic `torchao_linear_coverage`

### Model eligibility policy

Mostly reusable:

- `modules/sd_models.py`
  - `mxfp8_linear_region()` logic
  - region policy skip structure
  - VAE/other exclusion
  - conservative MultiheadAttention `out_proj` exclusion for LoRA safety

This should be factored into neutral helpers only after NVFP4 has passed smoke testing. First pass can duplicate with `nvfp4_*` names to avoid destabilizing MXFP8.

### High-level quantization transaction

Reusable conceptually:

- count eligible/policy-skipped/incompatible Linear modules
- temporarily detach `first_stage_model`
- store immutable CPU BF16 base weights for managed Linear modules
- clear active LoRA prepared state
- load quantized cache when valid
- otherwise `torchao.quantization.quantize_()` selected modules
- write stats onto the model

Current MXFP8 implementation:

- `modules/sd_models.py::apply_mxfp8_weight_quantization()`

NVFP4 should get its own `apply_nvfp4_weight_quantization()` first. The transaction shape is reusable; the exact config, tensor detection, cache, and stats names should diverge.

### Fresh-model reload discipline

Reusable:

- `modules/sd_models.py::DisableFastModelLoadingForMxfp8`
- MXFP8 mode-change handling in `reload_model_weights()`
- forced fresh reload when enabling/disabling/changing coverage
- checkpoint cache invalidation when TorchAO tensor-subclass modes are active

NVFP4 should be included in the same class of unsafe-to-overwrite modes, but the implementation should use neutral naming such as `torchao_quant_mode_changed` / `DisableFastModelLoadingForTorchAOQuant` once both modes exist.

### Device movement workaround

Reusable after generalization:

- `modules/sd_models.py::send_mxfp8_model_to_device()`

This currently skips `MXTensor` leaves because `nn.Module.to()` may call tensor-moving/aliasing ops unsupported by TorchAO tensor subclasses. NVFP4 uses `NVFP4Tensor`, so this must diverge until generalized.

Recommended target:

- `send_torchao_quant_model_to_device()`
- skip both `MXTensor` and `NVFP4Tensor`
- move ordinary parameters and buffers around those leaves

### LoRA active-config architecture

Reusable conceptually and very valuable:

- `extensions-builtin/Lora/networks.py::prepare_mxfp8_active_config()`
- `network_mxfp8_active_config_signature()`
- `network_apply_mxfp8_merged_lora()`
- hot-path guard in `network_Linear_forward()`

The core idea should be reused for NVFP4:

1. keep immutable BF16 CPU master weights
2. merge active LoRA deltas into BF16 effective weights once per active config
3. quantize final effective Linear weights once
4. keep sampling `Linear.forward()` as the normal fast path

But the attributes and quantize config should diverge initially so MXFP8 remains untouched.

## Codepaths that need to diverge

### Format config and technical constraints

Must diverge:

- current: `modules/mxfp8_config.py`
- proposed: `modules/nvfp4_config.py`

Reasons:

- MXFP8 uses `MXTensor` and `MXDynamicActivationMXWeightConfig` with float8 e4m3 data/scales.
- NVFP4 uses `NVFP4Tensor` and `NVFP4DynamicActivationNVFP4WeightConfig` with FP4 packed data, FP8 e4m3 local scales, per-tensor scale, block size 16, and optional MSLK/Triton activation scaling.
- MXFP4 uses `MXTensor` with `float4_e2m1fn_x2`, but is not currently viable on GB10.

Do not hide this behind `get_mxfp8_config(format="nvfp4")`; make the backend explicit.

### Tensor detection

Must diverge or be explicitly generalized:

- MXFP8 checks: `MXTensor`
- NVFP4 checks: `NVFP4Tensor`

Affected areas:

- config technical skip helpers
- model cache validation
- diagnostics
- device movement
- LoRA quantized-state checks

### Disk cache

Must diverge initially:

- current: `modules/mxfp8_model_cache.py`
- proposed: `modules/nvfp4_model_cache.py`

Reasons:

- different tensor subclass safe globals
- different sidecar config names
- different cache directory (`mxfp8` vs `nvfp4`)
- likely different shape/compat constraints and future static/dynamic scaling metadata

Do not reuse the same cache directory or sidecar suffix. The cache key must include format/backend and coverage.

### Diagnostics/API

Should diverge initially:

- current: `modules/mxfp8_diagnostics.py`
- current API: `scripts/mxfp8_diagnostics_api.py`

Options:

1. Add sibling NVFP4 diagnostics/API first.
2. Later refactor both into a generic `torchao_quant_diagnostics` surface.

The first pass should not weaken the known-good MXFP8 diagnostics.

### LoRA state attributes

Must diverge or be carefully generalized:

Current MXFP8 attributes include:

- `network_mxfp8_base_weight`
- `network_mxfp8_base_bias`
- `network_mxfp8_active_config_signature`
- `network_mxfp8_prepare_stats`
- `network_mxfp8_prepare_error`
- `network_mxfp8_merged_lora_applied`

NVFP4 can use sibling names:

- `network_nvfp4_base_weight`
- `network_nvfp4_base_bias`
- `network_nvfp4_active_config_signature`
- `network_nvfp4_prepare_stats`
- `network_nvfp4_prepare_error`
- `network_nvfp4_merged_lora_applied`

A later cleanup can generalize these to `network_torchao_quant_*`, but the safer first implementation is explicit sibling code.

### Mode mutual exclusion

Must be extended:

- current: native FP8 vs MXFP8 only
- target: native FP8 vs MXFP8 vs NVFP4, all mutually exclusive

Do not allow MXFP8 and NVFP4 to be active at the same time in the first implementation.

## Mixed MXFP8/NVFP4 per-layer policy

Technically possible later, but not a safe first step.

A mixed policy would need:

- one unified per-module backend selector, not one global mode
- cache entries carrying per-layer backend metadata
- LoRA preparation selecting the right quantize config per module
- device movement skipping multiple TorchAO tensor subclasses
- diagnostics reporting per-layer backend and quality/perf stats
- stronger active-config signatures including per-layer backend policy

Same-layer mixed operands are different from per-layer mixed backends:

- TorchAO `MXDynamicActivationMXWeightConfig` currently requires matching activation and weight dtype.
- It does not expose MXFP8 activations with MXFP4 weights as a public workflow config.
- MSLK has lower-level mixed symbols, but wiring those directly into A1111 would be a research project, not a toggle.

Recommendation: ship one global NVFP4 mode first, compare quality/speed/memory against MXFP8, then consider per-region/per-layer format policy only if NVFP4 proves useful.

## Recommended implementation order

1. Add `modules/nvfp4_config.py` and tiny runtime import/smoke validation.
2. Add `devices.nvfp4`, `nvfp4_storage`, and mutual exclusion with native FP8/MXFP8.
3. Add `apply_nvfp4_weight_quantization()` using the MXFP8 transaction shape but separate stats and cache.
4. Add `modules/nvfp4_model_cache.py` with `NVFP4Tensor` safe globals and `Stable-diffusion/nvfp4/` cache root.
5. Generalize device movement to skip both `MXTensor` and `NVFP4Tensor`.
6. Add NVFP4 LoRA active-config preparation by copying the MXFP8 merge-then-quantize design with NVFP4-specific attributes/config.
7. Add diagnostics/API coverage for NVFP4.
8. Validate in phases: no-LoRA txt2img, no-LoRA img2img, single LoRA txt2img, repeated-step timing, coverage changes, enable/disable reloads, cache load/save, log scan.

## Recommendation

Implement NVFP4 as a sibling backend, not as a thin MXFP8 toggle. Reuse the MXFP8 architecture and policies, but keep the codepaths separate where TorchAO tensor type, cache serialization, LoRA state, diagnostics, and reload handling touch stability-sensitive behavior.

Do not start with MXFP4 on GB10; current live tests show it is blocked by TorchAO/PyTorch hardware gating. Do not start with mixed MXFP8/NVFP4 per layer; it multiplies state/cache/LoRA risks before we know whether NVFP4 is worth using.
