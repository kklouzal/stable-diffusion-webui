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

After the initial refactor:

- `0` LoRAs: warm `2.541s`, repeat `2.059s`, repeat `0.515s/step`
- `1` LoRA: warm `32.497s`, repeat `2.096s`, repeat `0.524s/step`
- `4` LoRAs: warm `7.865s`, repeat `2.064s`, repeat `0.516s/step`
- `13` LoRAs: warm `13.922s`, repeat `2.094s`, repeat `0.523s/step`

Follow-up cleanup strengthened the prepared-state signature check, added whole-transaction rollback on prepare failure, includes MXFP8 config identity in the active signature, reports detailed prepare errors in the UI comment, prebuilds the quantization config once per prepare transaction, and prevents no-op active-LoRA modules from being left in stale BF16 state during forced prepare.

Post-cleanup validation on the hot-patched live container, same `unet_other` benchmark:

- `0` LoRAs: warm `2.183s`, repeat `2.052s`, repeat `0.513s/step`
- `1` LoRA: warm `31.060s`, repeat `2.042s`, repeat `0.510s/step`
- `4` LoRAs: warm `8.039s`, repeat `2.025s`, repeat `0.506s/step`
- `13` LoRAs: warm `13.428s`, repeat `2.061s`, repeat `0.515s/step`

The repeat generation path is now effectively flat across LoRA count, which validates the intended invariant.

An img2img smoke with `<lora:Detail-Enhancer-v1.0:0.6:0.6>`, `unet_other`, and merge-then-quantize completed successfully in `24.368s` for 4 steps after correcting the test payload to avoid its older BF16 override. Logs showed the expected prepare sequence: `183` managed Linear modules, `183` quantized, `0` kept BF16, `1` LoRA.

Live testing was restored to the conservative live safety state after benchmarking. This is not the product default; it intentionally disables MXFP8 Linear coverage on the running service until another explicit test/use pass enables it:

- `mxfp8_storage`: `Enable for SDXL`
- `mxfp8_linear_coverage`: `[]`
- `mxfp8_lora_mode`: `Merge LoRA then quantize to MXFP8`

## Final reload-path polish

A later end-to-end pass found that MXFP8 coverage transitions could still leave noisy meta-tensor load failures in the live logs, especially around SDXL conditioner/VAE reload paths. The root cause was the generic A1111 meta-device RAM optimization, which is useful for ordinary model loading but too fragile for MXFP8 reloads because custom SDXL/OpenCLIP/VAE state-dict paths can bypass the patched `Module._load_from_state_dict` route.

`modules/sd_models.py` now disables the fast/meta loader only while MXFP8 storage is requested, forcing fully materialized model construction/load for MXFP8 startup and coverage reloads. This is deliberately scoped to MXFP8 because those reloads are comparatively rare and correctness is more important than preserving the generic RAM optimization there.

`modules/mxfp8_diagnostics.py` also aliases `prepare_stats` as `mxfp8_active_config_stats`, so smoke tooling can directly report the active prepared LoRA/MXFP8 configuration instead of `null`.

Final validation from rebuilt image `sha256:05ab63a366b73ab6c4a69fc1969f54995410a790fe4034ba6dc0f9004a20a210`:

- startup fresh-log scan: no `Cannot copy out`, `failed to prepare`, `Traceback`, `RuntimeError`, or checkpoint loading errors
- img2img smoke: completed in `26.318s` for 4 steps with one active LoRA; diagnostics reported `prepared_linear=183`, `quantized_linear=183`, `failed_linear=0`, `active_lora_count=1`
- repeat benchmark: `0` LoRAs `2.045s` / `0.511s/step`; `1` LoRA `2.093s` / `0.523s/step`; `4` LoRAs `2.064s` / `0.516s/step`; `13` LoRAs `2.109s` / `0.527s/step`
- final live safety state restored to `mxfp8_storage=Enable for SDXL`, `mxfp8_linear_coverage=[]`, `mxfp8_lora_mode=Merge LoRA then quantize to MXFP8`
- fresh validation log scan: no `Cannot copy out`, `failed to prepare`, `Traceback`, `RuntimeError`, or checkpoint loading errors
