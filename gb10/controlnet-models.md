# GB10 ControlNet model setup

The `sd-webui-controlnet` extension is repo-owned under `extensions/sd-webui-controlnet` and `gb10/run.sh` mirrors it
into `/opt/gb10/stable-diffusion/Extensions/sd-webui-controlnet` on every deploy.

Put ControlNet models in the checkout: `~/stable-diffusion-webui/extensions/sd-webui-controlnet/models/`. The
weights are git-ignored (large data); commit a `<model file>.sha256` sidecar (`sha256sum <file>`, bare file name)
next to each one. run.sh copies them to the host `models/` directory, which it protects (`P /models/***`), so a
model placed only on the host also survives a deploy, but removing a model from the checkout does not remove it from
the host: delete it there by hand.

Every checkpoint in use is SDXL, so the models must be SDXL ControlNets. Installed today:

- `xinsir-controlnet-depth-sdxl-1.0.safetensors` (xinsir/controlnet-depth-sdxl-1.0): depth control, used with the
  `depth_zoe` preprocessor (and `depth_anything` / `depth_anything_v2`).
- `destitech-controlnet-inpaint-dreamer-sdxl-e3738810.fp16.safetensors` (destitech/controlnet-inpaint-dreamer-sdxl):
  inpaint control.

Preprocessor (annotator) weights are separate: they download on first use into the host
`Extensions/sd-webui-controlnet/annotator/downloads/`, which run.sh also preserves.
