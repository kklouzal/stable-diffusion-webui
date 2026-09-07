# Last generation API (schema version 2)

`GET /sdapi/v1/generation/last` returns the latest successfully completed A1111 generation. It uses the existing `/sdapi/v1` HTTP Basic authentication and is read-only: it never starts a generation, changes global options, or loads a model.

The endpoint returns HTTP `404` when no valid snapshot exists:

```json
{"detail":"No successfully completed generation snapshot is available"}
```

## Retention and durability

API extension arguments that exceed an extension's fixed UI slot use request-local isolated ranges. Execution and snapshot capture read the same range, preventing an oversized extension from overwriting neighboring settings. Fixed-slot prefixes remain available to extensions that read their original positions directly.

Exactly one snapshot is retained at `${HOST_ROOT}/config/generation-last/generation-last.json`, bind-mounted at `/opt/stable-diffusion-webui/generation-last/generation-last.json`. The temporary file is created in that same mounted directory, fsynced, atomically replaced with `os.replace`, then the directory is fsynced. A completed generation is captured under one process lock, so settings and image assets cannot mix across concurrent completions. Failed, cancelled, stopped, and image-less generations do not replace the retained record.

The current record survives container restarts and recreation. Version-1 txt2img records are sanitized in memory to this version-2 contract on read, so an existing retained txt2img record stays available immediately after upgrade. A version-1 img2img record is non-replayable because it could not contain its input assets; run one img2img generation to replace it.

Limits are 2 MiB per encoded image, 6 MiB for all encoded input assets, and 8 MiB for the JSON snapshot. Images exceeding a limit are not retained and the snapshot names the missing asset in `limitations`.

## Contract

Every successful response has these fields:

```json
{
  "schema_version": 2,
  "completed_at": "2026-09-06T20:00:00.000000Z",
  "generation_type": "txt2img",
  "replayable": true,
  "limitations": [],
  "checkpoint": {
    "title": "checkpoint title [hash]",
    "name": "checkpoint title",
    "hash": "short-hash",
    "sha256": "full-hash-or-null",
    "filename": "checkpoint.safetensors",
    "vae_name": "vae.safetensors-or-null",
    "vae_hash": "vae-hash-or-null"
  },
  "parameters": {}
}
```

`parameters` is a request object for the endpoint selected by `generation_type`: `/sdapi/v1/txt2img` or `/sdapi/v1/img2img`. It includes effective checkpoint/VAE overrides, sampler, scheduler, steps, CFG, denoising, seed/subseed controls, refiner/high-resolution controls, resize/inpaint controls, and serializable selectable and always-on script arguments. `token_merging_ratio` and `token_merging_ratio_hr` are retained; credential-like keys remain filtered.

The snapshot deliberately omits the previous positive prompt, negative prompt, root `width` and `height`, first-pass dimensions, high-resolution scale/resize dimensions, and high-resolution prompts. It also omits matching named values inside nested mapping settings. Harness supplies those values.

### Redacted txt2img example

```json
{
  "schema_version": 2,
  "generation_type": "txt2img",
  "replayable": true,
  "limitations": [],
  "parameters": {
    "sampler_name": "Euler",
    "scheduler": "Automatic",
    "steps": 20,
    "cfg_scale": 7.0,
    "seed": 123456789,
    "subseed": 123456789,
    "batch_size": 1,
    "n_iter": 1,
    "enable_hr": false,
    "override_settings": {"sd_model_checkpoint": "checkpoint title [hash]"},
    "alwayson_scripts": {"ControlNet": {"args": [{"enabled": false, "module": "none", "model": "None"}]}}
  }
}
```

### Redacted img2img example

```json
{
  "schema_version": 2,
  "generation_type": "img2img",
  "replayable": true,
  "limitations": [],
  "parameters": {
    "init_images": ["<base64-encoded PNG>"],
    "mask": "<base64-encoded PNG>",
    "denoising_strength": 0.55,
    "resize_mode": 0,
    "inpaint_full_res": true,
    "inpaint_full_res_padding": 32,
    "inpainting_mask_invert": 0,
    "sampler_name": "Euler",
    "scheduler": "Automatic",
    "steps": 20,
    "cfg_scale": 7.0,
    "alwayson_scripts": {
      "ControlNet": {
        "args": [{"enabled": true, "module": "canny", "model": "control-model", "image": "<base64-encoded PNG>", "mask": "<base64-encoded PNG>"}]
      }
    }
  }
}
```

Enabled ControlNet units retain supported API fields and their API-base64 `image` and `mask` inputs when within limits. Both UI unit objects and API argument dictionaries use the same bounded image serializer, so replaying an API snapshot does not discard its encoded inputs. Inline base64 and PNG/JPEG/WebP image data URIs are decoded locally; paths and URLs are never opened. Disabled units do not retain unused images. Effective-region masks, IP-Adapter serialized inputs, and batch inputs that cannot be safely retained are never replaced with defaults: the snapshot is `replayable: false` and names the exact missing `alwayson_scripts.ControlNet.args[n]` field in `limitations`.

## Harness replay and overrides

Harness must require `schema_version == 2`, `replayable == true`, and a supported `generation_type`. Start with `snapshot.parameters`, then supply both replacement prompts and these mandatory overrides before posting to the matching endpoint:

```json
{
  "prompt": "Harness-provided positive prompt",
  "negative_prompt": "Harness-provided negative prompt",
  "seed": -1,
  "subseed": -1,
  "subseed_strength": 0.0,
  "seed_resize_from_h": 0,
  "seed_resize_from_w": 0,
  "width": 1024,
  "height": 1024,
  "batch_size": 1,
  "n_iter": 1,
  "enable_hr": false,
  "firstphase_width": 0,
  "firstphase_height": 0,
  "hr_scale": 1.0,
  "hr_upscaler": null,
  "hr_second_pass_steps": 0,
  "hr_resize_x": 0,
  "hr_resize_y": 0,
  "hr_checkpoint_name": null,
  "hr_sampler_name": null,
  "hr_scheduler": null,
  "hr_prompt": "",
  "hr_negative_prompt": "",
  "refiner_checkpoint": null,
  "refiner_switch_at": 1.0,
  "send_images": true,
  "save_images": false,
  "override_settings_restore_afterwards": true
}
```

Do not replace or shorten positional `script_args` or `alwayson_scripts.*.args`: their indices are extension contracts. Apply extension-specific values at their documented positions. Any extension that can alter prompts, dimensions, seeds, batches, high-resolution passes, refiner selection, or image inputs must be explicitly disabled with that extension's own API arguments, or those positional arguments must be updated in place. If the extension has no API disable/override argument, Harness must reject replay as non-comparable. For the generic 1024x1024 run, clearing an entire always-on script is only safe when that extension accepts an empty argument list; otherwise preserve positions and set its documented disabled values.

For img2img, keep retained `init_images`, `mask`, and enabled ControlNet image/mask assets in the final request. Do not substitute filesystem paths: all retained images are API base64 PNG strings.

Example shell flow:

```bash
curl --fail --user "$A1111_USER:$A1111_PASSWORD" \
  http://127.0.0.1:7860/sdapi/v1/generation/last > last.json

jq '.parameters + {
  prompt: "Harness-provided positive prompt", negative_prompt: "Harness-provided negative prompt",
  seed: -1, subseed: -1, subseed_strength: 0,
  seed_resize_from_h: 0, seed_resize_from_w: 0,
  width: 1024, height: 1024, batch_size: 1, n_iter: 1,
  enable_hr: false, firstphase_width: 0, firstphase_height: 0,
  hr_scale: 1, hr_upscaler: null, hr_second_pass_steps: 0,
  hr_resize_x: 0, hr_resize_y: 0, hr_checkpoint_name: null,
  hr_sampler_name: null, hr_scheduler: null, hr_prompt: "", hr_negative_prompt: "",
  refiner_checkpoint: null, refiner_switch_at: 1,
  send_images: true, save_images: false, override_settings_restore_afterwards: true
}' last.json | curl --fail --user "$A1111_USER:$A1111_PASSWORD" \
  -H 'Content-Type: application/json' --data-binary @- \
  "http://127.0.0.1:7860/sdapi/v1/$(jq -r .generation_type last.json)"
```
