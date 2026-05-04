# OpenClaw Clear Cond Cache

First-class GB10/A1111 fork extension that exposes a small local API endpoint to clear AUTOMATIC1111 prompt-conditioning caches while leaving `persistent_cond_cache` enabled.

Endpoint:

- `POST /sdapi/v1/openclaw/clear-cond-cache`

The endpoint resets:

- `StableDiffusionProcessing.cached_c`
- `StableDiffusionProcessing.cached_uc`
- `StableDiffusionProcessingTxt2Img.cached_hr_c`
- `StableDiffusionProcessingTxt2Img.cached_hr_uc`
