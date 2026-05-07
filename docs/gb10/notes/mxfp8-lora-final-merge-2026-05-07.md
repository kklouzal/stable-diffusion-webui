# MXFP8 final-merged LoRA preparation — 2026-05-07

## Problem

MXFP8 `unet_other` generation stayed fast with no active LoRAs, but repeat generation time scaled almost linearly with active LoRA count after the MXFP8 LoRA merge/requantize work:

- `0` LoRAs: ~`0.51s/step`
- `1` LoRA: ~`0.82s/step`
- `4` LoRAs: ~`1.43s/step`
- `8` LoRAs: ~`2.20s/step`
- `13` LoRAs: ~`3.2s/step`

That shape proved the LoRA/MXFP8 integration was still doing LoRA-count-sensitive work in the generation path rather than preparing one final effective in-memory weight configuration.

## Intended invariant

For a given checkpoint, MXFP8 coverage setting, LoRA set, LoRA strengths, dyn dim values, and LoRA file identity:

1. rebuild MXFP8-managed Linear weights from immutable BF16 master weights;
2. apply active LoRA deltas once;
3. quantize the final effective weights once when the selected LoRA mode requests MXFP8 final weights;
4. mark the model and all managed Linear modules prepared for that exact signature;
5. run `Linear.forward()` as a pure fast path until that signature changes.

There is intentionally no disk cache for LoRA permutations. The cache is one active in-memory prepared configuration.

## Implementation

`extensions-builtin/Lora/networks.py` now treats MXFP8+LoRA preparation as a model-level transaction:

- `prepare_mxfp8_active_config()` computes the active signature and prepares all MXFP8-managed Linear modules outside the sampler hot path.
- `network_apply_mxfp8_merged_lora(..., force=True)` is used by the prepare transaction to rebuild from `network_mxfp8_base_weight` / `network_mxfp8_base_bias` rather than from current mutable weights.
- Managed `network_Linear_forward()` no longer performs per-layer LoRA scanning, merging, quantization, or functional fallback during sampling. If the model is not prepared, it prepares the whole active config once or raises rather than silently reintroducing the slow per-step fallback.
- `ExtraNetworkLora.activate()` now treats MXFP8 active-config preparation failure as a visible generation-stopping error.

`modules/sd_models.py` now annotates MXFP8-managed layers with their FQN/region and clears stale prepared signatures when MXFP8 base quantization is applied.

`modules/mxfp8_diagnostics.py` exposes prepared-config stats/error state when available.

## Validation

Rebuilt image: `local/gb10-a1111:latest-mxfp8-dev` (`sha256:067427ac2bb5684f9dc65d244194b197f73b7c0ce647a03177e28f1b613a914a`). Live container relaunched as `gb10-a1111-latest-mxfp8` from that image.

Benchmark payload: `832x832`, `4` steps, `Euler a`, `Karras`, `test2.safetensors`, MXFP8 storage `Enable for SDXL`, coverage `['unet_other']`, mode `Merge LoRA then quantize to MXFP8`.

After the refactor:

- `0` LoRAs: warm `2.541s`, repeat `2.059s`, repeat `0.515s/step`
- `1` LoRA: warm `32.497s`, repeat `2.096s`, repeat `0.524s/step`
- `4` LoRAs: warm `7.865s`, repeat `2.064s`, repeat `0.516s/step`
- `13` LoRAs: warm `13.922s`, repeat `2.094s`, repeat `0.523s/step`

The repeat generation path is now effectively flat across LoRA count, which validates the intended invariant.

Live testing was restored to the safe state after benchmarking:

- `mxfp8_storage`: `Enable for SDXL`
- `mxfp8_linear_coverage`: `[]`
- `mxfp8_lora_mode`: `Merge LoRA then quantize to MXFP8`
