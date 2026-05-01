# GB10 img2img path notes

## Low-risk cleanup boundary

For GB10 A1111 img2img work, keep cleanup/refactors outside final image math unless a dedicated deterministic image-regression pass is planned. Safe work includes API task lifecycle cleanup, API schema/docs, response-shaping tests, and idempotent resource cleanup. Avoid changing mask/crop/latent-noise/seed/sampler/denoising behavior in a general cleanup pass.

## API response knobs

- `init_images`: required list of base64/data-URI images for `/sdapi/v1/img2img`.
- `mask`: optional base64/data-URI inpaint mask.
- `include_init_images`: controls whether `parameters.init_images` and `parameters.mask` are echoed back in the response; it does not affect generation.
- `send_images`: controls whether generated images are included as base64 in the response; it does not affect generation.
- `save_images`: controls whether generated images are saved to disk; it does not affect generation.

## SDXL manual smoke shape

Prefer SDXL-representative smoke payloads over tiny SD1.5-style fixtures when validating GB10 runtime behavior:

- 1024x1024-ish input/init size
- low step count for smoke speed
- known SDXL checkpoint/VAE/runtime defaults
- `send_images=false` and `save_images=false` when timing core generation rather than response/save overhead
- fixed seed and sampler for comparisons

Keep existing generic 64x64 tests as API-shape coverage only; do not treat them as GB10 SDXL quality coverage.
