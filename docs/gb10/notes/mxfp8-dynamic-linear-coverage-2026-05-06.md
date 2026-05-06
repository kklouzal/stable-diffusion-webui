# MXFP8 dynamic Linear coverage — 2026-05-06

This note records the experimental runtime-selectable MXFP8 Linear coverage control added after the final conservative img2img baseline.

## Runtime option

`MXFP8 Linear coverage` is a checkbox-group optimization option with four supported Linear regions:

- `unet_other`
- `self_attention`
- `cross_attention`
- `conditioner`

The default remains `unet_other`, matching the conservative baseline documented in `mxfp8-img2img-final-baseline-2026-05-06.md`.

Changing the selection triggers a forced model-weight reload, so the newly selected coverage set is applied from BF16 checkpoint weights and active LoRA state is reapplied through the normal merge-then-quantize path. This is intentionally more conservative than trying to mutate an already-quantized live module graph in place.

## Unsupported regions

TorchAO `MXDynamicActivationMXWeightConfig` in the current GB10 runtime supports the MXFP8 inference workflow for `torch.nn.Linear`; it does not provide a matching Conv2d MXFP8 handler.

Therefore these native-A1111-FP8-style regions remain out of scope for this MXFP8 control:

- `conv unet_other`
- `conv vae`

VAE remains skipped. Conv MXFP8 would require a separate implementation path or upstream TorchAO support, not just a policy toggle.

## Expected coverage steps

For the current SDXL checkpoint/module graph, the important Linear coverage levels are expected to be approximately:

- `unet_other`: `183` Linear modules
- `unet_other + self_attention`: `463` Linear modules
- `unet_other + self_attention + cross_attention`: `743` Linear modules
- all four regions including `conditioner`: about `911` Linear modules

Use these counts as quick sanity checks in `/sdapi/v1/mxfp8-diagnostics` after each coverage reload.
