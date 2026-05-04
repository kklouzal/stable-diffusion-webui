# OpenClaw Controller Helpers

First-class GB10/A1111 fork extension that exposes small local API endpoints used by A1111-Controller.

Endpoints:

- `POST /sdapi/v1/openclaw/clear-cond-cache`
- `GET /sdapi/v1/openclaw/cond-cache`
- `POST /sdapi/v1/openclaw/token-count`
- `POST /sdapi/v1/openclaw/token_counter` compatibility alias

The cond-cache endpoint resets:

- `StableDiffusionProcessing.cached_c`
- `StableDiffusionProcessing.cached_uc`
- `StableDiffusionProcessingTxt2Img.cached_hr_c`
- `StableDiffusionProcessingTxt2Img.cached_hr_uc`

The token-count endpoint accepts JSON `{"text": string, "steps": number}` and returns `{"ok": true, "token_count": number, "max_length": number}` using A1111's active tokenizer/model hijack path after stripping extra-network tags and expanding prompt schedules.
