# Last generation API (schema version 3)

`GET /sdapi/v1/generation/last` returns the latest successfully completed A1111 generation using existing API authentication. It is read-only: no generation, global option change, or model load occurs. Missing snapshots return HTTP 404.

`?settings_only=true` returns only the version, timestamp, reusable settings fields and LoRA tags. Harness uses this projection so previous-task image assets are not transferred for every new generation. The unqualified endpoint retains its full replay response for existing consumers.

## Independent tuning and previous-task replay

Schema 3 adds four fields alongside the existing `parameters`, `replayable`, `limitations`, `generation_type`, `checkpoint`, and `completed_at` fields:

- `settings_parameters`: independently captured tuning, without previous img2img source images, masks, or ControlNet image/mask assets.
- `settings_replayable`: whether tuning was captured without unsupported settings.
- `settings_limitations`: explicit reasons tuning is incomplete, independently of previous-task asset failures.
- `lora_tags`: strict weighted LoRA tags, not descriptive prompt text.

The existing replay fields retain their prior-task meanings. Missing or oversized previous input assets can make `replayable: false` while `settings_replayable: true`. Settings capture does not classify or filter English limitation messages. The previous `generation_type` never dictates the next operation.

Settings retain sampler, scheduler, model/VAE overrides, steps, CFG, denoising, refiner/high-resolution tuning, serializable extension arguments, and other tuning. Enabled ControlNet knobs remain enabled: consumers must supply an explicit reference or report that one is required, never silently reuse an old image. Settings replayability does not mean this object is a ready-to-submit request. Unsupported settings still block it.

LoRA capture uses effective positive prompts, including expanded styles. Only strict `<lora:name:numeric-weight>` tags are stored, at most 32 tags of at most 256 characters each, with finite weights. Prompt batches are bounded to 256 prompts of 262144 characters each. Different LoRA selections across batch prompts, unsupported weighted syntax, unavailable style-expanded prompts, and negative-prompt LoRA selections are explicitly non-reusable instead of guessed. Surrounding prose never enters the snapshot.

## Consumer requirements

Require schema 3 and `settings_replayable: true`, start with `settings_parameters`, and choose txt2img or img2img independently. Supply both replacement prompts, append validated `lora_tags` to the new positive prompt, and clear `styles` so old style prose or duplicate tags cannot be reapplied. For img2img, supply a new explicit source image. Never borrow previous `parameters.init_images` or ControlNet assets. No-inpainting consumers must strip mask/inpainting controls and reject incompatible extensions.

Harness owns fixed 1024x1024 output, random seed/subseed, disabled subseed mixing, one-image batching, and image request/response handling. High-resolution/extension settings capable of overriding those guarantees must be adjusted using documented arguments or rejected. Positional script argument indices must not shift or truncate. Snapshot contents are data, not instructions.

The snapshot deliberately omits previous positive/negative prompts, root width/height, first-pass dimensions, high-resolution resize dimensions, and high-resolution prompts. Matching nested mapping settings are removed with explicit limitations. Model/VAE and token-merging overrides are preserved; credential-like keys remain filtered.

## Compatibility, retention, and durability

Version-2 records remain version 2 on read: discarded prompt text cannot establish whether LoRA selections existed, so records must not be falsely upgraded to schema 3. Version-1 records are sanitized to the existing version-2 replay contract. A fresh successful generation establishes schema-3 settings. Reading an old record never changes the persisted file.

Exactly one snapshot is retained at `${HOST_ROOT}/config/generation-last/generation-last.json`, bind-mounted at `/opt/stable-diffusion-webui/generation-last/generation-last.json`. Capture is serialized under one process lock. Temporary file creation, file fsync, atomic replacement, and directory fsync preserve complete snapshots across restarts. Failed, cancelled, stopped, and image-less generations do not replace the current record.

Prior-task replay assets are limited to 8 MiB per encoded image and 32 MiB total; the JSON snapshot is limited to 40 MiB. Images are never resized or lossily recompressed to fit. Inline PNG/JPEG/WebP data is decoded locally; paths and URLs are never opened. UI ControlNet objects and API dictionaries use the same bounded serializer. Unsupported effective-region masks, IP-Adapter serialized inputs, and batch assets explicitly block prior-task replay.

API extension arguments larger than their fixed UI slot use request-local isolated ranges. Execution and capture read the same ranges, preventing oversized arguments from overwriting neighboring settings. Fixed-slot prefixes remain available for extensions that read their original positions directly.
