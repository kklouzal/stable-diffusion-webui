# GB10 ControlNet model setup

The `sd-webui-controlnet` extension is repo-owned under `extensions/sd-webui-controlnet` and `gb10/run.sh` mirrors it
into `/opt/gb10/stable-diffusion/Extensions/sd-webui-controlnet` on every deploy.

Put ControlNet weights in `/models/local/stable-diffusion/controlnet/models/`.
Keep each `<model file>.sha256` sidecar beside its selected model. `gb10/run.sh`
binds this directory at the extension's existing in-container `models/` path;
weights no longer live in the source checkout or mirrored extension directory.
The central store's ownership and ACLs are preserved during deployment.

Every checkpoint in use is SDXL, so the models must be SDXL ControlNets. Installed today:

- `xinsir-controlnet-depth-sdxl-1.0.safetensors` (xinsir/controlnet-depth-sdxl-1.0): depth control, used with the
  `depth_zoe` preprocessor (and `depth_anything` / `depth_anything_v2`).
- `destitech-controlnet-inpaint-dreamer-sdxl-e3738810.fp16.safetensors` (destitech/controlnet-inpaint-dreamer-sdxl):
  inpaint control.

Preprocessor (annotator) weights are separate: they download on first use into the host
`/models/local/stable-diffusion/controlnet/annotator-downloads/`, bound at the
extension's original in-container `annotator/downloads/` path.
