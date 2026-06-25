# GB10 ControlNet model setup

The `sd-webui-controlnet` extension is repo-owned under `extensions/sd-webui-controlnet` and is mirrored by `gb10/run.sh` into `/opt/gb10/stable-diffusion/Extensions/sd-webui-controlnet` before starting `gb10-a1111-latest`.

Default outline/edge workflow uses the ControlNet `canny` preprocessor. Put a compatible model in:

`/opt/gb10/stable-diffusion/Extensions/sd-webui-controlnet/models/`

Recommended source: `lllyasviel/ControlNet-v1-1` on Hugging Face, file `control_v11p_sd15_canny.pth` or `.safetensors` if available. The file is large model data and is intentionally not committed to git.
