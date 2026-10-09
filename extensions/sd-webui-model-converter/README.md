# OpenClaw Model Converter

Fork-local, OpenClaw-owned A1111 extension for single-checkpoint and LoRA conversion.

This extension was adopted from `Akegarasu/sd-webui-model-converter` at upstream commit `a8c04410aa505be61652dca1ba6361bd85113667` (`float8_e5m2 (#29)`, 2024-12-24), then trimmed and maintained as part of Schwi's GB10 A1111 fork.

## Supported workflow

- Single-checkpoint conversion from the A1111 checkpoint list
- Precision conversion: `fp32`, `fp16`, `bf16`, `float8_e4m3fn`, `float8_e5m2`
- Pruning: disabled, no-EMA, EMA-only
- Output formats: `safetensors`, `ckpt`
- Optional VAE bake-in. The VAE list is re-read before each conversion; an unknown VAE name, or a bake-in with the VAE action `delete`, fails the request
- Per-weight-family action: convert, copy, or delete for UNet, CLIP/text encoder, VAE, and other weights. The `v_pred` and `ztsnr` marker keys, which A1111 reads as model configuration, are copied even when other weights are deleted
- Range-checked precision: a weight the target precision cannot hold (for example above 65504 for `fp16` or 57344 for `float8_e5m2`) fails the conversion with its key named, instead of becoming Inf (which the NaN/Inf repair would zero) or saturating (`float8_e4m3fn`). Upstream cast without a check
- NaN/Inf scan and repair (to 0) of every floating-point tensor, before and after conversion, reported in the doctor report
- CLIP `position_ids` int64 preservation/fix
- Known junk-data prefix removal
- LoRA conversion (`mode: "lora"`): a LoRA from A1111's LoRA listing re-saved as safetensors in `fp32`, `fp16` or `bf16`, with NaN/Inf repair, optional training-residue cleanup and a LoRA doctor report; the source's content-hash metadata (`sshs_model_hash`, `sshs_legacy_hash`, `modelspec.hash_sha256`) is not copied
- OpenClaw API endpoints for A1111-Controller integration

Converted checkpoints are written next to the source checkpoint, matching the original extension behavior; converted LoRAs are written next to the source LoRA. An output never replaces an existing file: it is written to a hidden temporary file, fsynced, and hard-linked to its final name (which fails if the name exists), then the directory is fsynced. The model directory must be on a filesystem with hard links.

## API

- `GET /sdapi/v1/openclaw/model-converter/options`
- `POST /sdapi/v1/openclaw/model-converter/convert`

The controller normally runs conversion in its own background worker and calls the POST endpoint, so long conversions do not block the controller UI.
