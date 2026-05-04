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
- MSLK: PyTorch nightly `cu130`; currently `2026.5.2+cu130`

MSLK note: as of this change, a CUDA 13.2 MSLK wheel was not available from the selected PyTorch nightly lane, while the CUDA 13.0 aarch64 nightly wheel was available and validated on GB10. The source-build lane remains available if the wheel stops aligning with GB10 hardware or if a source build proves better.

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
- MSLK source build may become useful, but the available PyTorch nightly `cu130` aarch64 wheel already validates the Triton NVFP4 path on GB10.
