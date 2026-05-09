# OpenClaw Model Converter

Fork-local, OpenClaw-owned A1111 extension for single-checkpoint conversion.

This extension was adopted from `Akegarasu/sd-webui-model-converter` at upstream commit `a8c04410aa505be61652dca1ba6361bd85113667` (`float8_e5m2 (#29)`, 2024-12-24), then trimmed and maintained as part of Schwi's GB10 A1111 fork.

## Supported workflow

- Single-checkpoint conversion from the A1111 checkpoint list
- Precision conversion: `fp32`, `fp16`, `bf16`, `float8_e4m3fn`, `float8_e5m2`
- Pruning: disabled, no-EMA, EMA-only
- Output formats: `safetensors`, `ckpt`
- Optional VAE bake-in
- Per-weight-family action: convert, copy, or delete for UNet, CLIP/text encoder, VAE, and other weights
- CLIP `position_ids` int64 preservation/fix
- Known junk-data prefix removal
- OpenClaw API endpoints for A1111-Controller integration

Converted checkpoints are written next to the source checkpoint, matching the original extension behavior.

## API

- `GET /sdapi/v1/openclaw/model-converter/options`
- `POST /sdapi/v1/openclaw/model-converter/convert`

The controller normally runs conversion in its own background worker and calls the POST endpoint, so long conversions do not block the controller UI.
