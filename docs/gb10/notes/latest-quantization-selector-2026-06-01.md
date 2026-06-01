# Unified latest quantization selector

Date: 2026-06-01
Branch: `latest`
Base: `latest-mxfp8` at `4ba907111876ba222f8edd19dcc11a1698678b39`

## Goal

`latest` is the unified GB10 branch for the mature MXFP8 runtime plus selectable NVFP4. `latest-mxfp8` remains the untouched known-good branch.

MXFP8 stays the default backend. NVFP4 is exposed as a sibling backend selected by runtime options rather than as a replacement for the MXFP8 implementation.

## Port audit

The old `latest-nvfp4` branch has 17 commits not present as commits on `latest-mxfp8`, but most useful NVFP4 capability has already been reimplemented on top of the newer MXFP8 architecture:

- protected TorchAO/MSLK image foundation
- `--dtype bfloat16` runtime support
- `nvfp4_storage` and `nvfp4_linear_coverage`
- mutual exclusion across native FP8, MXFP8, and NVFP4
- separate NVFP4 config and model cache
- fresh TorchAO reload handling for MXFP8/NVFP4 mode and coverage changes
- NVFP4 LoRA merge-then-quantize active-config preparation
- controller-side `mxfp8` / `nvfp4` backend selection

The old branch pieces intentionally not ported are:

- global BF16 safetensors asset cache for checkpoints, LoRAs, and VAE files
- old VAE NVFP4 quantization experiment
- old API-only attention backend route

Those paths broaden model loading behavior and are not required for selectable NVFP4. Leaving them out avoids disturbing the current MXFP8 quality and performance baseline.

## Added in this branch

- `/sdapi/v1/openclaw/precision-map`
- compatibility alias `/sdapi/v1/precision-map`
- branch-default image tag `local/gb10-a1111:latest`
- branch-default container name `gb10-a1111-latest`

The precision map reports active MXFP8/NVFP4 tensor subclasses, selected coverage, LoRA targets, per-layer skip reasons, quantization stats, and cache hits. It is an inspection surface for validating runtime quantization switches without modifying generation behavior.

## Runtime switch shape

A1111 option changes for `mxfp8_storage`, `mxfp8_linear_coverage`, `nvfp4_storage`, and `nvfp4_linear_coverage` synchronously run `reload_model_weights(forced_reload=True)`.

The reload path treats active or requested MXFP8/NVFP4 as unsafe to overwrite in place, clears checkpoint state-dict cache, detaches the old TorchAO-mutated model, reloads from disk, applies the requested quantization backend, and leaves LoRA active-config preparation for the next generation signature.

A1111-Controller applies quantization options before generation, then applies compile/cuDNN/CUDA-graph settings. A settings-signature change clears CUDA graphs.
