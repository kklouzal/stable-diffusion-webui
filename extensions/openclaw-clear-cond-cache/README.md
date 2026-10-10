# OpenClaw Controller Helpers

First-class GB10/A1111 fork extension that exposes small local API endpoints used by A1111-Controller, and a backend activity status lane fed by hooks on A1111's model, VAE and LoRA loading functions.

Endpoints:

- `POST /sdapi/v1/openclaw/clear-cond-cache`
- `GET /sdapi/v1/openclaw/cond-cache`
- `POST /sdapi/v1/openclaw/token-count`
- `POST /sdapi/v1/openclaw/token_counter` compatibility alias of `token-count`
- `POST /sdapi/v1/openclaw/torch-compile`
- `GET /sdapi/v1/openclaw/torch-compile`
- `GET /sdapi/v1/openclaw/backend-status`
- `POST /sdapi/v1/openclaw/cudnn-benchmark`
- `GET /sdapi/v1/openclaw/cudnn-benchmark`
- `POST /sdapi/v1/openclaw/model-merge`
- `GET /sdapi/v1/openclaw/training-templates`

The clear-cond-cache endpoint runs under A1111's `queue_lock`. Without a body it clears every reusable generation cache: the conditioning slots `StableDiffusionProcessing.cached_c`/`cached_uc` and `StableDiffusionProcessingTxt2Img.cached_hr_c`/`cached_hr_uc`, and the img2img init-latent cache. An optional `targets` (or `target`) field selects a subset: a name, a list of names, or a `{name: bool}` map, where the names are `all`, `prompt`, `base`, `hr`, `img2img` (with a few synonyms) or a slot name (`c`, `uc`, `hr_c`, `hr_uc`, `img2img_init`); unknown names are ignored. Clearing conditioning slots bumps the conditioner and conditioning-hook epochs in the same epoch transaction. The response lists the `cleared` slots and the normalized `targets`. `GET .../cond-cache` reports which slots are populated and the img2img init cache status.

The token-count endpoint accepts JSON `{"text": string, "steps": number}` and returns `{"ok": true, "token_count": number, "max_length": number}` using A1111's active tokenizer/model hijack path after stripping extra-network tags and expanding prompt schedules; it reports the count of the longest scheduled prompt. While no text encoder is loaded it returns `ok: false` with no count.

`POST .../torch-compile` with `{"vae": bool}` (`enabled` and `"target": "vae-only"` are accepted too) never compiles: it echoes the `requested` value and answers `status.vae: false` with `disabled_reason.vae` "VAE decode uses CUDA graphs; module compile does not reach decode/encode". `GET` returns the same status. The VAE compile it replaced wrapped `first_stage_model` with `torch.compile`, which compiled only `forward` (the core calls `decode`/`encode`) and renamed the VAE's state_dict keys to `_orig_mod.*`, so an in-place checkpoint switch skipped every VAE weight.

`POST .../cudnn-benchmark` with `{"enabled": bool}` sets `torch.backends.cudnn.benchmark` under `queue_lock`, so it never changes between the steps of one image; `GET` returns the current value.

`POST .../model-merge` runs A1111's checkpoint merger (`extras.run_modelmerger`) under `queue_lock` with the merger's fields (`primary_model_name`, `secondary_model_name`, `tertiary_model_name`, `interp_method`, `multiplier`, `checkpoint_format`, `save_as_half`, metadata options and so on). `GET .../training-templates` lists the textual-inversion template names.

Boolean body fields are parsed, not truth-tested: a value that is not a boolean returns `{"ok": false, "error": ...}`.

`GET .../backend-status` returns the innermost active backend phase (checkpoint load/read/apply, model creation and device move, empty-prompt conditioning, VAE load, LoRA loading, quantized-weight preparation, startup model load) with its label and detail, or `active: false` when idle.
