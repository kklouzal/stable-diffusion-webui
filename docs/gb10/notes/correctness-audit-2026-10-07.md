# Correctness and API-currency audit — 2026-10-07

Scope: every SDXL txt2img/img2img code path of `local/gb10-a1111:latest` (core, owned extensions, host-mounted
extensions via deploy patchers, API boundary), audited for correct use of CUDA/PyTorch/numerics and libraries, and for
current torch 2.14 APIs. Items are labelled **exact** (bitwise unchanged), **fix** (behaviour change that corrects a
defect; output changes where stated) or **contract** (API/request behaviour change). Nothing was made faster at the
cost of correctness. All verification was CPU-only: GPU work needs an explicit go-ahead on this host
(see "GPU verification still owed").

Full evidence (ledger, per-area findings, test logs, probes):
`~/audit-artifacts/gb10-a1111-correctness-20261007/` (LEDGER.md, findings/A..G, R1..R3, t-*.txt).

## SEG report ("looks like a sign is flipped")

No code regression was found. A CPU differential of the previous deploy (492d2c8d) against the audited tree, with the
real CFG denoiser, SEG, Dynamic Thresholding, ControlNet (balanced and cfg-injection), PAG, TeaCache, the real prompt
parser and the "Multi: oi" chain in img2img, gives SEG effects with cosine +1.0000 in fp32 and +0.1..+0.4 (never
negative) under bf16 autocast; only uncond rows are blurred in both trees and the infinite-blur path is bit-identical.
The guidance D = U + s(C - U) + (s - 1)(U - U') has the right sign, and Dynamic Thresholding only rescales by
non-negative factors. The likely cause is the sampler: "Multi: oi" (`Euler a[Exponential] -> DPM++ 2M SDE[AYS]`) never
ran for these img2img requests before this deploy (before 2026-09-27 the name fell back to the default sampler; from
then until the multi-sampler splice fix every such request produced an all-NaN latent), so no earlier SEG image used it.
Settling experiment (GPU): fixed seed, PAG off — "Multi: oi" with SEG off/on, and DPM++ 2M + AYS with SEG off/on.

## Contract changes

- Alwayson script hooks and per-step/output callbacks (`process`, `process_batch`, `before_hr`, cfg_denoiser/denoised/
  after_cfg, extra_noise, before_image_saved, ...) fail the request with the hook and script named, instead of
  rendering without the extension (production logs showed ControlNet `process` failures answered 200 without
  ControlNet). `postprocess_batch`/`postprocess` also run after a failed generation for every script that ran a hook,
  so per-request hooks are always removed. Startup/UI/notification callbacks stay log-only.
- A missing or malformed `<lora:...>` tag, a non-finite LoRA multiplier, or any extra-network activation error fails
  the request (before: every LoRA was dropped silently).
- `/options` and `override_settings` validate value types and report failed keys (422/500; `override_settings` is
  checked before the job starts); settings, checkpoint, training and runtime-switch endpoints wait for `queue_lock`,
  as do `/controlnet/detect`, `/controlnet/model_list?update=true` and `/sdapi/v1/refresh-loras` (they ran blocking
  work on the event loop and replaced registries a generation was reading); graph switches and clear-cond-cache body
  flags parse booleans strictly (`"false"` meant True); malformed infotext values answer 422.
- ControlNet `advanced_weighting` of the wrong length raises (was silently truncated).
- model-converter accepts only listed checkpoints/LoRAs (raw paths allowed arbitrary file read/write over the API).
- SEG requested on a model without middle-block self-attention raises.

## Output-changing fixes

- Decoded images round to uint8 (IEC 61966-2-1 round(255 E')); truncation biased every pixel by -0.5 code.
- SDXL VAE encodes return the posterior mean (fp32): the sample came from the unseeded global CPU generator, so img2img
  init latents depended on earlier requests (removed noise <= 6e-5, median 3e-6).
- LoRA deltas are summed in fp32 and rounded once (stacked LoRAs were 22-65% of elements off the fp64 reference), merges
  run outside bf16 autocast; DoRA axis, lora_A/B alpha, IA3 conv axis, GLoRA scale, OFT bf16, fp8 fp16-cache stacking.
- Original emphasis restores the mean in fp32 (exact identity at multiplier 1; the bf16 mean rescaled every chunk by up
  to 2^-9); clip_l tokenization applies ftfy.fix_text again (transformers 5 dropped it; clip_g's open_clip applies it);
  textual inversion vectors move with their tokens across comma backtracking.
- Schedulers: KL Optimal ends at sigma 0; Simple and Align Your Steps match their float64 references (AYS moves <= 16
  float32 ulp); exact floors for img2img step counts, prompt-editing switch steps and hires target size.
- Latent blend masks are float32 (soft inpaint masks were quantized to bf16).
- ControlNet: inpaint/colorfix eps<->x0 post-processing in fp32; hires-pass detection by both dimensions; hires conds
  marked (hires rows were all read as cond); hires hint size follows the core's hires latent; IP-Adapter picks image
  k/v per call row (uncond reused cond k/v); PuLID/plus-SDXL uncond follow their references.
- SEG blurs exactly the uncond rows of the full CFG batch (AND prompts / batch>=2 cond-only calls blurred cond rows) and
  derives the attention grid from the UNet downsample chain (wrong for ~47% of sizes that are not multiples of 64).
- Face restoration rounds; SwinIR/ScuNET/LDSR run fp32 outside hires autocast; upscalers hit their exact target size.
- Hypertile non-square row/column order.

## Crashes and latent defects fixed (no change on the production path)

- Under the deployed bf16 VAE, ControlNet reference/adain, inpaint_only(+lama), tile_colorfix and the SD-inpaint hijack
  always failed (numpy has no bfloat16); live previews and save-before-hires crashed on bf16 `.numpy()`.
- 1-step DPM++ 2M/3M SDE schedules, UniPC logSNR, SDXL-inpaint latent hires conditioning, ScuNET untiled device,
  Anyline resolution, ControlNet hires resize with a 0 dimension.
- `upcast_attn` was a no-op under CUDA autocast on every SDPA path (incl. the Tiled VAE patch).
- CUDA/VAE graph keys gained the installed attention forward, SDPA backend, upcast_attn and lora_functional; graphs
  bypass hypernetworks; TorchAO master restores re-prepare LoRA quantization.
- TeaCache never restores the UNet underneath a live ControlNet wrapper; ControlNet low-VRAM hooks register once.
- A failed extras request no longer leaves the "extras" job active; `reload_model_weights` is serialized with every
  other model load (`model_data.lock`); `errors.display` records the exception it shows.
- The fail-closed hook contract was reviewed against every installed external extension (Detail Daemon, Tiled
  Diffusion/VAE, demofusion, Ultimate SD Upscale) and the in-tree scripts: none raises in normal operation, and their
  cleanup tolerates the empty failure shapes. Known pre-existing: Ultimate SD Upscale raises if interrupted before its
  first tile; Detail Daemon's uncond mode scales cond rows for AND prompts.
- Security: Pillow pixel limit no longer disabled process-wide by X/Y/Z; EXIF parsed by Pillow; API image URLs check
  every redirect hop.

## Performance side effects (exact)

LoRA file digests memoized per file revision (0.1-13.7 s per activation), CUDA-graph schedule signature interned
(~1.5 ms host per denoiser call), depth_zoe BEiT index kept on device (48 host-to-device copies + syncs per call),
ControlNet schedule tables on device, DW-pose ONNX parsed once, ControlNet models rebuilt on checkpoint switches only
when they are 'difference' models.

## Open decisions

- Two reviewed-but-unmerged branches: API image-URL byte/pixel budgets with connections pinned to the checked address
  (`worktree-agent-a3b67454aa3b8f5d9`, 5952302e) and atomic, verified gb10 patcher writes
  (`worktree-agent-a6047407e608c950c`, d7777073).
- Saving a custom multi-sampler swaps sampler entries without `queue_lock` (a microsecond window for a running
  generation); taking the lock would make the controller's 30 s timeout fire behind long generations.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` (fragmentation on shared unified memory) needs a GPU measurement.

## GPU verification still owed

1. SEG settling experiment above.
2. One request per alwayson extension (ControlNet depth_zoe, PAG, SEG, Dynamic Thresholding, TeaCache, multi-sampler,
   Detail Daemon, Tiled Diffusion/VAE, hypertile) to confirm none raises in normal use under the fail-closed contract;
   a deliberately failing request (Dynamic Thresholding + UniPC) followed by a normal one matching a fresh-process image.
3. Image A/B of the output-changing fixes on the captured workloads; LoRA merge time with fp32 math.
4. The live-server API tests (test/test_txt2img.py, test_img2img.py, test_extras.py, test_utils.py).
