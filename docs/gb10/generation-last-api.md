# Last generation API

`GET /sdapi/v1/generation/last` returns one durable snapshot of the most recently completed A1111 generation. It uses the same HTTP Basic authentication as the rest of `/sdapi/v1`.

The endpoint is read-only. It does not run a generation, load a checkpoint, or modify A1111 options. A missing, unreadable, incomplete, or unsupported snapshot returns HTTP `404` with:

```json
{"detail":"No successfully completed generation snapshot is available"}
```

The snapshot is atomically written to `generation-last/generation-last.json`. GB10 bind-mounts the whole directory from `${HOST_ROOT}/config/generation-last` at `/opt/stable-diffusion-webui/generation-last`; the temporary file is created and replaced in that same mounted directory. The write fsyncs both the file and its directory before returning, so it remains atomic and durable across container replacement or restart. A failed, skipped, interrupted, stopped, or image-less generation cannot replace the existing snapshot.

## Response contract

```json
{
  "schema_version": 1,
  "completed_at": "2026-09-06T12:00:00.000000Z",
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
  "parameters": {
    "prompt": "effective prompt",
    "negative_prompt": "effective negative prompt",
    "seed": 123456789,
    "sampler_name": "Euler",
    "scheduler": "Automatic",
    "steps": 20,
    "cfg_scale": 7.0,
    "width": 512,
    "height": 512,
    "batch_size": 1,
    "n_iter": 1,
    "enable_hr": false,
    "override_settings": {
      "sd_model_checkpoint": "checkpoint title [hash]",
      "sd_vae": "vae.safetensors",
      "CLIP_stop_at_last_layers": 2
    },
    "alwayson_scripts": {
      "Extension title": {"args": [true, 0.25]}
    }
  }
}
```

`parameters` contains A1111 API field names. A `txt2img` snapshot with `replayable: true` can be posted directly to `/sdapi/v1/txt2img`. It contains the resolved seed, effective sampler and scheduler, high-resolution controls, checkpoint/VAE override settings, ControlNet settings other than images, and replayable selectable/always-on script arguments.

The endpoint never stores image or mask payloads. Therefore every `img2img` snapshot is marked non-replayable and names the missing `init_images` and, where relevant, mask. A txt2img snapshot with enabled ControlNet is also non-replayable until the caller supplies each omitted `control_net_image`, `control_net_image2`, or `control_net_image3`. An unsupported or oversized parameter is not silently dropped: it is listed in `limitations` and makes the snapshot non-replayable.

## Harness replay request

The harness should first require `generation_type == "txt2img"` and `replayable == true`. Start from `snapshot.parameters`, then apply these overrides before posting to `/sdapi/v1/txt2img`:

```json
{
  "prompt": "a red apple on a wooden table, studio photograph",
  "negative_prompt": "",
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
  "script_name": null,
  "script_args": [],
  "alwayson_scripts": {},
  "send_images": true,
  "save_images": false,
  "override_settings_restore_afterwards": true
}
```

The seed/subseed and seed-resize overrides prevent variation or resized-noise seeds from replacing the requested `-1` seed. The high-resolution and refiner overrides prevent a second denoise pass, alternate checkpoint, alternate sampler, or alternate scheduler from changing the 1024x1024 request. Clearing selectable and persisted always-on script arguments prevents snapshot extension arguments from rewriting prompts, seeds, dimensions, batch count, or high-resolution controls.

Always-visible extensions that have no API disable argument still execute at their runtime default; no generic A1111 request can disable such an extension. A harness must use the extension's documented disabled arguments if it enables those extensions at startup, or reject the replay as non-comparable.

Example shell flow:

```bash
curl --fail --user "$A1111_USER:$A1111_PASSWORD" \
  http://127.0.0.1:7860/sdapi/v1/generation/last > last.json

jq '.parameters + {
  prompt: "a red apple on a wooden table, studio photograph",
  negative_prompt: "", seed: -1, subseed: -1, subseed_strength: 0,
  seed_resize_from_h: 0, seed_resize_from_w: 0,
  width: 1024, height: 1024, batch_size: 1, n_iter: 1,
  enable_hr: false, firstphase_width: 0, firstphase_height: 0,
  hr_scale: 1, hr_upscaler: null, hr_second_pass_steps: 0,
  hr_resize_x: 0, hr_resize_y: 0, hr_checkpoint_name: null,
  hr_sampler_name: null, hr_scheduler: null, hr_prompt: "", hr_negative_prompt: "",
  refiner_checkpoint: null, refiner_switch_at: 1,
  script_name: null, script_args: [], alwayson_scripts: {},
  send_images: true, save_images: false, override_settings_restore_afterwards: true
}' last.json | curl --fail --user "$A1111_USER:$A1111_PASSWORD" \
  -H 'Content-Type: application/json' --data-binary @- \
  http://127.0.0.1:7860/sdapi/v1/txt2img
```
