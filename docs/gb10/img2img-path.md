# GB10 img2img path notes

## Reference records

Read these before you change image-affecting math, quantization coverage, attention backend defaults, LoRA
quantization behavior, or SEG/PAG semantics.

Current records:
- [`notes/correctness-audit-2026-10-07.md`](notes/correctness-audit-2026-10-07.md): the image-affecting fixes and their
  live verification.
- [`notes/performance-pass-2-2026-10-07.md`](notes/performance-pass-2-2026-10-07.md): the exact (bitwise) performance
  changes and the parked layout work.

Historical record, from 2026-05-06:
- [`notes/mxfp8-img2img-final-baseline-2026-05-06.md`](notes/mxfp8-img2img-final-baseline-2026-05-06.md) is the MXFP8
  img2img baseline. Production now runs with `mxfp8_storage` and `nvfp4_storage` set to `Disable`, which is also the
  default.
- Runtime-selectable Linear coverage is in
  [`notes/mxfp8-dynamic-linear-coverage-2026-05-06.md`](notes/mxfp8-dynamic-linear-coverage-2026-05-06.md).

## Low-risk cleanup boundary

For GB10 A1111 img2img work, keep cleanup/refactors outside final image math unless a dedicated deterministic image-regression pass is planned. Safe work includes API task lifecycle cleanup, API schema/docs, response-shaping tests, and idempotent resource cleanup. Avoid changing mask/crop/latent-noise/seed/sampler/denoising behavior in a general cleanup pass.

## API response knobs

- `init_images`: required, non-empty list of base64/data-URI images for `/sdapi/v1/img2img` (missing: 404, empty: 422).
- `mask`: optional base64/data-URI inpaint mask. Its alpha is the mask when it has transparency; a mask of another size
  is stretched onto the init image.
- Input images: 16-bit grayscale is rounded to 8 bits, and RGB/CMYK images with a non-sRGB ICC profile are converted to
  sRGB. I/F-mode images (undefined value range) and malformed or mismatched ICC profiles answer 422.
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
