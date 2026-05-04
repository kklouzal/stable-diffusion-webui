# NVFP4 / FP4 experimental route for GB10 A1111

Date: 2026-05-04
Repo: `~/stable-diffusion-webui`, branch `latest`
Scope: planning and implementation notes for adding an experimental TorchAO NVFP4 path to the GB10 AUTOMATIC1111 fork.

## Current conclusion

NVFP4 is the practical experimental route. Plain `torch.float4_e2m1fn_x2` is not enough by itself.

The current GB10 PyTorch nightly exposes FP4 shell dtypes, but direct float4 casts and normal tensor ops are not a working model-storage/runtime path. TorchAO's NVFP4 tensor path supplies the missing quantized tensor wrapper, scaling, and kernel dispatch machinery.

## Verified baseline before integration

Observed against `local/gb10-a1111:base-protected-app-latest` before the TorchAO/MSLK image change:

- PyTorch: `2.13.0.dev20260502+cu132`
- CUDA as reported by torch: `13.2`
- CUDA capability on GB10: `(12, 1)`
- Exposed torch dtypes:
  - `torch.float4_e2m1fn_x2`
  - `torch.float8_e8m0fnu`
  - `torch.float8_e4m3fn`
- Missing from the previous image:
  - `torchao`
  - `mslk`
  - `nvidia-modelopt`
  - `torch-tensorrt`
  - `diffusers`

Direct `torch.float4_e2m1fn_x2` casting failed:

- CPU: `NotImplementedError "copy_kernel" not implemented for 'Float4_e2m1fn_x2'`
- CUDA: `RuntimeError copy_() does not support casting Float4_e2m1fn_x2 to different types`

TorchAO transient install tests succeeded for bf16 `torch.nn.Linear`:

- `torchao==0.17.0`
- `mslk==2026.5.2+cu130`
- `NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=True, use_triton_kernel=True)`
- bf16 Linear input/weight path produced finite bf16 output on GB10.

TorchAO NVFP4 failed for fp16:

- `AssertionError torch.float16 not supported`

So end-to-end bf16 support is a prerequisite for the real A1111 runtime path.

## Package integration decision

TorchAO/MSLK belong in the protected `torch-base` layer rather than in ordinary A1111 app dependencies.

Reason:

- TorchAO is tightly coupled to torch dtype/tensor-subclass behavior.
- MSLK supplies the Triton NVFP4 kernel path that TorchAO calls.
- App dependency resolution must not replace or shadow the CUDA/PyTorch/Triton/quantization substrate.

Current package lane:

- torch/torchvision/torchaudio: PyTorch nightly `cu132`
- TorchAO: `torchao` from the active Python package index; currently `0.17.0`
- MSLK: source-built from `https://github.com/meta-pytorch/MSLK.git` at `952739ea2f527b2fe776e025eaec983bda9d394d`; currently `2026.5.4`

MSLK note: the earlier CUDA 13.0 aarch64 nightly wheel (`2026.5.2+cu130`) validated the TorchAO NVFP4 Triton path, but GB10 now uses a source-built MSLK baseline against the active CUDA 13.2 / PyTorch nightly stack. This keeps `mslk.so` aligned for the bf16/NVFP4 implementation work.

## Why this is not checkpoint conversion first

The tempting mental model is:

1. Convert checkpoint weights to FP4/NVFP4.
2. Convert LoRA weights to FP4/NVFP4.
3. Load everything normally.

That is probably the wrong first implementation for A1111.

A safer first model is:

1. Load normal checkpoint weights.
2. Move selected model components to bf16.
3. Apply TorchAO NVFP4 quantization in memory to eligible `torch.nn.Linear` layers.
4. Keep Conv/VAE/text/fragile layers unquantized at first.
5. Add persistent serialization only after runtime behavior and quality are understood.

## A1111 code areas likely involved

### `Dockerfile`

Completed foundational change:

- Install TorchAO/MSLK in `torch-base`.
- Freeze them into `/opt/build/base-python-protected-constraints.txt` and `/opt/build/base-python-protected-names.txt`.
- Keep later app installs from replacing them.

### `docker/render-build-manifest.py`

Completed foundational change:

- Classify `torchao` and `mslk` as `Base-Provided|Torch-Quantization-Stack` in image manifests.

### `gb10/smoke-test.sh`

Completed foundational change:

- Require `torchao` and `mslk` imports.
- Run a small CUDA bf16 Linear NVFP4 TorchAO/MSLK smoke using the Triton path.

### `modules/devices.py`

Future work:

- Add explicit NVFP4/FP4 mode flags.
- Add real bf16 model/inference dtype support instead of forcing the current fp16/fp32-centered choices.
- Ensure autocast/manual-cast behavior does not silently fight bf16/NVFP4.

### `modules/shared_options.py`

Future work:

- Add an experimental option such as `fp4_storage` or `nvfp4_storage`.
- Initial option should be conservative, e.g. `Disable` / `NVFP4 Linear-only experimental`.

### `modules/sd_models.py`

Future work:

- Add a post-load quantization pass after `load_state_dict()` and after model dtype placement.
- Quantize only selected `torch.nn.Linear` layers.
- Skip VAE at first.
- Skip text encoder at first unless separately validated.
- Skip small, input, output, embedding, and norm-adjacent layers until quality is profiled.
- Avoid storing TorchAO quantized tensors in normal checkpoint cache assumptions until verified.

### `extensions-builtin/Lora/networks.py`

Future work:

Existing LoRA fast path mutates base weights in-place:

- stores base weight backups
- computes LoRA up/down delta
- does `weight + updown`
- writes back with `copy_()`

That breaks on TorchAO `NVFP4Tensor` because common tensor ops and copy semantics are not implemented like normal dense tensors.

MVP LoRA strategy:

- when NVFP4 is active, force functional LoRA or add a quant-aware runtime delta path
- compute `base_forward(x)` through quantized Linear
- compute LoRA delta in bf16
- add outputs without modifying the quantized base weight

Do not quantize LoRA weights initially. The memory win is small and compatibility risk is high.

## Recommended implementation phases

### Phase 0 — package/kernel foundation

Status: in progress with this note/change.

- Add TorchAO/MSLK to protected torch layer.
- Validate imports and a tiny NVFP4 Triton Linear smoke.

### Phase 1 — bf16 foundation

- Add end-to-end bf16 model/inference support.
- Verify current fp16 behavior remains unchanged by default.
- Verify bf16 generation without NVFP4 first.

### Phase 2 — NVFP4 no-LoRA proof

- Add experimental NVFP4 Linear-only mode.
- Quantize conservative UNet/transformer Linear subset only.
- Generate comparison images with no LoRA.
- Record memory, latency, and visual drift.

### Phase 3 — functional LoRA compatibility

- Force or implement functional LoRA when NVFP4 is active.
- Keep LoRA weights bf16.
- Validate normal prompt-triggered LoRA selection and generation.

### Phase 4 — performance/quality tuning

- Tune filter function by layer names/shapes.
- Try `torch.compile` only after the eager path is stable.
- Consider persistent quantized artifact caching if load-time quantization is too slow.

## Current risk notes

- NVFP4 needs bf16; fp16 failed in direct tests.
- A1111's current dtype model is mostly fp16/fp32-oriented.
- Existing LoRA weight mutation is incompatible with TorchAO NVFP4 tensors.
- Diffusers benchmark wins rely on selective quantization, TorchAO, MSLK/Triton, and often `torch.compile`; A1111 will not inherit those speedups automatically.
- MSLK is now source-built against the active CU132 PyTorch lane; the older `cu130` wheel remains only a historical fallback.

## 2026-05-04 native-option MVP validation

First native-style NVFP4 option path is now validated in the A1111 fork.

Implemented surfaces:

- `modules/shared_options.py`: added `NVFP4 weight` option beside `FP8 weight`, with `Disable` / `Enable for SDXL` / `Enable` choices.
- `modules/devices.py`: added runtime `devices.nvfp4` state beside `devices.fp8`.
- `modules/initialize_util.py`: option changes force model reload from checkpoint weights.
- `modules/sd_models.py`: added model-load guardrails, FP8/NVFP4 mutual exclusion, bf16/CUDA requirement checks, selective TorchAO quantization, and load-time reporting.
- `modules/processing.py` and `modules/infotext_utils.py`: generation metadata now records/restores `NVFP4 weight` similarly to `FP8 weight`.

Initial quantization path:

- Uses `torchao.quantization.quantize_` with `torchao.prototype.mx_formats.NVFP4DynamicActivationNVFP4WeightConfig`.
- Applies to eligible `torch.nn.Linear` modules whose in/out feature dimensions are divisible by 16.
- Temporarily excludes `first_stage_model`/VAE during the pass.
- Requires `--dtype bfloat16`; float16 is rejected because TorchAO NVFP4 direct tests fail there.
- Leaves FP8 Conv/Linear weight storage as a separate mutually-exclusive path.

Validation image/runtime:

- Built image: `local/gb10-a1111:nvfp4-native-test`.
- Runtime container: `gb10-a1111-latest` with `--dtype bfloat16`.
- Options: `fp8_storage=Disable`, `nvfp4_storage=Enable for SDXL`, `cache_fp16_weight=False`.
- Model: `test2.safetensors` (`sd_xl_base.yaml`).
- Load log: `Applied NVFP4 weight quantization to 911 Linear modules; skipped 0 incompatible Linear modules`.
- Load timing: `Model loaded in 10.2s (... apply nvfp4: 2.1s ...)`.
- Generation: 1024x1024, 20 steps, Euler, CFG 7, seed `123456789`.
- API wall time: `10.918s`.
- Sampler progress: final `20/20` at about `2.24 it/s`; steady-state mostly `2.27–2.30 it/s` after first step.
- Output: `/tmp/gb10_nvfp4_test2_1024.png` on GB10.

Current caveats:

- This validates first-stage eager NVFP4 generation, not LoRA compatibility.
- Text encoders remain included if their Linear shapes are eligible; the VAE is excluded.
- Next tuning should compare layer subsets, image drift, memory, LoRA behavior, and whether dynamic activation NVFP4 or weight-only NVFP4 is better for A1111.


## 2026-05-04 LoRA compatibility research/prototype

A1111's default LoRA path is not compatible with TorchAO `NVFP4Tensor` weights when the affected layer is quantized.

Observed failure before the compatibility patch:

- Prompt LoRA selection works and existing SDXL LoRA files are discovered normally.
- Generation fails when LoRA touches a quantized `torch.nn.Linear`.
- The first failure happened in `extensions-builtin/Lora/networks.py` while `network_apply_weights()` called `store_weights_backup(self.weight)` for a CLIP `q_proj` layer.
- TorchAO rejected the backup/copy operation with an assertion from its `__torch_dispatch__` path because `NVFP4Tensor.to(devices.cpu, copy=True)` is not a normal dense tensor copy.
- Forcing A1111's existing `lora_functional=True` was not sufficient while text-encoder Linear modules were quantized, because functional/non-functional behavior still reaches quantized tensor operations in fragile places.

Best first integration:

1. Keep LoRA checkpoint weights in normal bf16/fp16 tensors. Do not quantize LoRA files during initial LoRA load.
2. Quantize only the denoiser/UNet Linear path for now; leave SDXL text encoders and VAE unquantized.
3. Route LoRA over quantized Linear layers through A1111's functional LoRA forward path automatically, without globally forcing `lora_functional` for every layer.
4. Let the quantized base Linear run through TorchAO/MSLK, compute the LoRA delta separately in normal tensor math, then add the output delta. Avoid modifying/restoring/copying `NVFP4Tensor` base weights.

Prototype changes:

- `modules/sd_models.py`: `nvfp4_linear_filter()` now skips `conditioner.*`, `cond_stage_model.*`, and `first_stage_model.*`, and counts eligibility using `model.named_modules()` so load logs reflect the actual filtered set.
- `extensions-builtin/Lora/networks.py`: `network_Linear_forward()` detects TorchAO `NVFP4Tensor` weights and automatically uses `network_forward()` for those layers, even when the global `lora_functional` option is disabled.

Validation:

- Built image: `local/gb10-a1111:nvfp4-lora-test2`.
- Test container: `gb10-a1111-nvfp4-lora-test` on port `7861` with `--dtype bfloat16`.
- Options: `sd_model_checkpoint=test2.safetensors`, `fp8_storage=Disable`, `nvfp4_storage=Enable for SDXL`, `cache_fp16_weight=False`, `lora_functional=False`.
- LoRA prompt tags: `<lora:Detail-Enhancer-v1.0:0.6>`, `<lora:Canopus-Realism-LoRA:0.35>`.
- Load log: `Applied NVFP4 weight quantization to 743 Linear modules; skipped 168 incompatible Linear modules`.
- Load timing: `Model loaded in 9.0s (... apply nvfp4: 1.7s ...)`.
- Generation succeeded at 1024x1024, 20 steps, Euler, CFG 7, seed `424242424`.
- API wall time including first LoRA/model work: `38.281s`.
- Sampler steady-state after the first step was about `1.1 it/s` with two LoRAs active.
- Output: `/tmp/gb10_nvfp4_lora_patch_1024.png` on GB10, copied to workspace `tmp/gb10_nvfp4_lora_patch_1024.png`.

Current caveats:

- This is a compatibility-first path, not the final performance-tuned path.
- Leaving text encoders unquantized reduces NVFP4 coverage from the earlier 911 Linear modules to 743, but avoids the LoRA/text-encoder copy failure and is safer for prompt conditioning quality.
- Functional LoRA over quantized Linear layers costs performance; the first validated two-LoRA render was slower than no-LoRA NVFP4.
- Next profiling should compare no-LoRA, one-LoRA, and normal bf16 LoRA at matched prompts/seeds before deciding whether to optimize LoRA delta computation or selectively re-enable safe text-encoder quantization.

## Accepted benchmark/quality comparison settings

Schwi accepted the NVFP4 + LoRA `DPM++ 2M SDE` render quality on 2026-05-04 and asked to use the same settings and seed for testing/benchmarking moving forward so quality comparisons remain meaningful.

Baseline comparison settings:

- Model: `test2.safetensors`
- Sampler: `DPM++ 2M SDE`
- Schedule type: `Exponential`
- Steps: `20`
- Size: `1024x1024`
- CFG scale: `7`
- Seed: `424242424`
- Runtime dtype: `--dtype bfloat16`
- NVFP4 option: `Enable for SDXL`
- FP8 option: `Disable`
- LoRA comparison set from the accepted render:
  - `<lora:Detail-Enhancer-v1.0:0.6>`
  - `<lora:Canopus-Realism-LoRA:0.35>`

Use this as the standard quality/benchmark prompt configuration unless the comparison target specifically requires a different layer subset, LoRA set, or seed.
