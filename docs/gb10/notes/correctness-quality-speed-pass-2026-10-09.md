# Correctness, quality and speed pass — 2026-10-09

## Scope

The user asked for an end-to-end pass over the whole source for correctness, generation quality and generation speed,
iterated until no valid opportunity remained.

- **Base.** `3042c2c4` (deploy10 `c22a9794` plus docs).
- **Method.** Seven read-only area audits:
  - A: sampling and the processing core
  - B: model runtime
  - C: extra networks and conditioning caches
  - D: API, images and IO
  - E: ControlNet
  - F: owned extensions, deploy patchers and the third-party extensions they patch, including Tiled VAE
  - G: build, deploy and runtime configuration

  Every finding was re-verified before it was fixed. Nine implementation streams, each in its own worktree, were then
  merged into `audit-integration`, which was fast-forwarded into `latest`. Finding ids (A-01, C-01, ...) are used in
  the commit messages.
- **Exclusions.** Items the earlier notes list as fixed, rejected, parked or kept were not re-reported.

## Evidence

### CPU gate

The CPU gate is the image test suite run offline.

- Base: core 1433 passed, 2 failed, 235 skipped.
- Final tree: core 1715+ passed and the same 2 failures, every extension suite passing.
- The 2 failures are `test_face_restorers` gfpgan/codeformer: their model download needs the network.

### GPU A/B

The A/B ran on a separate server on port 7861, with production stopped, using the operator's last request:

- **n-w1:** img2img at 1280², "Multi: oi2" + Align Your Steps 15 steps, denoise 0.66, 8 LoRAs, ControlNet depth_zoe
  (xinsir), PAG 6 + SEG, Dynamic Thresholding, Detail Daemon, TeaCache, fixed seed and a neutral prompt.
- **plain:** txt2img at 1024², DPM++ 2M + AYS 15 steps, the same LoRAs.

Both images ran with identical mounts and flags, alternating. Round 2 of the timing session (idle host, 5 timed
requests each) gave:

| Build | plain | n-w1 |
|---|---|---|
| deploy10 | 4.138 s ± 0.016 | 11.335 s ± 0.023 |
| this pass | 4.131 s ± 0.015 | 11.299 s ± 0.036 |

- **Speed:** parity. The fixes cost nothing measurable.
- **Determinism:** every build was deterministic run to run.
- **Peak allocated memory:** 16.4 GB vs 14.7 GB, mostly the fp32 text encoders (L-01). Reserved memory is lower once
  expandable segments are on (below).

### Profile (nsys, warm n-w1 on deploy10)

- 11.8 s of kernel time in a 12.1 s request, so the request is GPU-bound.
- 14.8% of GPU time is strided layout copies around GroupNorm under `--opt-channelslast`. About 1.1 s of that is at
  the VAE's full-resolution sizes.
- The rest: GEMMs about 35%, flash attention about 10%, GEGLU 6.7%.
- The layout work stays parked by owner decision ("VAE-only layout" below).

### Final validation

The final validation ran on the real `build.sh` image of `c77391d7`. It has provenance labels and no dependency drift
from deploy10.

- **Workloads:** plain 4.126 s ± 0.003 and n-w1 11.259 s ± 0.023, against deploy10's 4.138 and 11.335. The pixels are
  identical to the A/B build.
- **Tiling:** tiling and plain requests in either order give identical images.
- **Graphs vs eager:** UNet/VAE graph output is bit-identical to eager for plain, tiling and Tiled VAE (fast and
  non-fast). The fused Tiled VAE normalization runs on CUDA.
- **Live API tests:** 35/35.
- **Ultimate SD Upscale:** 2048², 16 tiles of 512, padding 32, blur 8.
  - Graphs are bypassed on both builds.
  - Each tile is now generated at its exact crop size (616 px) instead of being resampled to 576 px and back. That is
    14% more pixels, and the request goes from 24.7 s to 28.3 s.
  - The owner chose the sharper exact-size tiles.

## Output changes on the production path

All of these were checked on the GPU. Sample sheets were shown to the operator.

- **PAG (F-01).** The perturbed pass is `to_out(to_v(x))`, as in the paper, diffusers' `PAGIdentitySelfAttnProcessor`
  and ComfyUI. Upstream Incantations returned raw `to_v(x)` and skipped the output projection. On n-w1 the subject is
  unchanged but the background differs.
- **Text-encoder LoRA keys (new finding, C stream).**
  - transformers 5's `CLIPTextModel` has no `text_model` submodule, so every CLIP-L LoRA key was dropped silently.
  - 4 of the 8 production LoRAs had lost all 72 CLIP-L modules (216 keys).
  - The diffusers-keyed SDXL upsampler key was mapped with the SD1 layout (C-01). The DMD2 LoRA lost a rank-64 conv
    delta.
  - All 8 production LoRAs now match every key.
- **Text encoders in fp32 (L-01).** MM_R2_FIX stores the conditioner in fp32, but it ran in bf16. Error against
  float64 on the real clip_g:

  | Precision | Rel L2 error |
  |---|---|
  | bf16 (before) | 3.3e-2 |
  | fp32 with autocast off and IEEE matmul (now) | 2.1e-5 |

  The cost is +1.5 GiB and no measurable time. Plain txt2img keeps its composition and changes in detail.
- **VAE input rounding.**
  - Decode (S-01): the latent is scaled in fp32 and rounded to bf16 once. Against an fp32-VAE decode of the same latent,
    PSNR is 44.01 dB vs 42.86 dB and mean error 0.79 vs 0.90 codes.
  - Encode (S-02): `x*2-1` is computed in fp32 and rounded once. Proven exact over all 256 codes.
- **Depth maps (E-06).** Rounded instead of truncated to uint8: at most 1 code on depth_zoe. Production inputs at
  multiples of 64 are not affected by E-04.

## Fixes by area (summary; details in commit messages)

### A: sampling and prompts

- **Prompt parsing:**
  - A trailing `16:9` / `10:30` is no longer a composable weight: a colon right after a digit is never a weight
    separator (A-01).
  - BREAK inside brackets is a chunk break (A-06).
  - Original emphasis restores each row's own mean (A-13).
- **Conditioning:**
  - The empty-prompt cond is recomputed after an in-place checkpoint switch (A-04).
  - A zeroed SDXL negative is padded with zeros (A-05).
- **Samplers:**
  - DDIM/DDIM CFG++/PLMS run their final step (A-12).
  - Request s_churn/s_tmin/s_tmax/s_noise take precedence and are recorded; Euler a, DPM adaptive and Restart honour
    s_noise (A-08, A-10).
  - The NaN check covers the whole tensor (A-11).
- **Multi-sampler:**
  - hires stages use their own scheduler (A-02);
  - one Brownian tree per chain (A-07);
  - the shared s_* resolver, with schedule infotext kept (A-09);
  - `snapshot_dir` is confined to its root (security).
- **img2imgalt** works on SDXL (A-16).
- **Exact speed items:**
  - one CPU sigma copy (A-17);
  - the refiner check syncs only before the switch (A-19).

### B: model runtime

- **Tiling and graph keys (B-01).** Tiling (circular padding) is in the UNet and VAE graph keys. A tiling request used
  to replay zero-padding graphs, and the reverse.
- **Alternative UNets (B-05)** bypass graphs and switch inside the mutation boundary.
- **In-place VAE switch (B-04).** VAE switches load in place on unified memory, with no 7 GB CPU round trip.

### C: LoRA, textual inversion, hypernetworks

- **Fail closed (C-02).**
  - A LoRA that cannot be merged (shape mismatch, merge error) fails the request.
  - Unmatched keys are reported in a `Lora errors` infotext entry.
- **Merging and caching:**
  - Partial q/k/v MHA LoRAs apply (C-04).
  - `lora_functional` switches republish and drop graphs (C-03).
  - Parsed networks belong to one model (C-05).
  - Only changed layers are re-merged (C-06).
  - The device cache is released only after an eviction (C-07).
  - Factors keep the file dtype (C-08).
- **Lookup and infotext:**
  - The LoRA list is published by swap (C-13).
  - `Lora hashes` uses the request's alias (C-11).
  - `dyn < 1` is rejected (C-14).
- **Textual inversion** reloads on changes anywhere in its folders, deterministically (C-10).
- **Hypernetworks** fail closed (C-12).
- **Conditioning cache** keys on the text-encoder LoRA state only (C-09, exact).

### D: API, images, IO

- **Masks and geometry:**
  - Resize mode 3 with a mask (D-01).
  - A mask of a different size from the init image (D-02).
  - Alpha in LA/PA/P masks (D-04).
- **Image decoding:**
  - 16-bit grayscale is rounded to 8 bits; I/F modes answer 422 (D-03).
  - ICC profiles are converted to sRGB; untagged and sRGB-tagged inputs stay exact (D-05).
- **img2img init cache** keeps preset color corrections out (A-03).
- **Latent-noise fill** uses each batch's seeds (A-14).
- **generation_last:**
  - It is written after the extra-network cleanup succeeds (D-10).
  - ASCII JSON, 78 -> 14 ms (D-07).
  - No PNG re-decode, 32.5 -> 2.6 ms (D-08).
- **`/progress`** never decodes on the polling thread (D-11, B-03).
- **`init_images: []`** answers 422 (D-18).
- **Extras:**
  - Alpha is kept through upscalers and face restore (D-12).
  - SwinIR/ScuNET tiles cross-fade (D-13).
  - A single resample to the exact size (D-14).
  - No mid-request `torch_gc` (D-15).
  - SwinIR uses the file-identity cache (D-16).
  - The extras cache does no work at size 0 (D-17).

### E: ControlNet

- **Depth Anything:**
  - v1/v2 get RGB input (E-01). This is a channel swap that was in upstream.
  - v1 builds DINOv2 from the vendored code, not `torch.hub` (E-02).
  - Depth preprocessors run on the unpadded image (E-04).
- **Inpaint masks** are prepared and cropped like the core (E-03).
- **Map resizing:** gray maps are never nearest-scaled (E-05).
- **Requests** no longer unload the unused preprocessors (E-07).
- **Hook:** no-op residual multiplies are skipped (E-08, bitwise-tested over 128 cases).

### F: owned extensions, patchers, Tiled VAE

- **PAG/SEG:**
  - The start/end steps compare the denoiser call's own step (F-07).
  - SEG blurs the uncond rows of every CFG call layout, on the latent's own grid (F-03, F-08).
  - SEG fails under Tiled Diffusion (F-02).
- **Dynamic Thresholding** wraps the hires sampler and keeps chain step counts (F-05, F-06).
- **Hypertile:**
  - the configured tile size (H3);
  - cropped hires (H1);
  - an O(1) enabled check (H4).
- **Model converter:**
  - `bake_in_vae` validation;
  - the v_pred markers are kept;
  - overflow is rejected;
  - an `isfinite` fast path;
  - no-clobber via a hard link.
- **MultiDiffusion:**
  - upstream tile origins with only the last one pinned (M1);
  - noise-inversion conditioning and cache (M3, M4);
  - stable ControlNet tiles (M5);
  - SDXL region control fails early (M6).
- **Ultimate SD Upscale:**
  - interrupts survive (U1);
  - overrides are applied once (U2);
  - exact-size tiles (U3, U4), one size for every tile of a pass (edge tiles shift inward), so a pass is one CUDA graph shape;
  - infotext (U5).
- **Tiled VAE:**
  - GroupNorm statistics from valid regions, pooled exactly (T1). PSNR vs untiled goes from 47.1 to 59.0 dB.
  - Normalization with fp32 statistics, rounded once (T2).
  - One NaN check per tile (T3).
  - Encoder tiles on the latent grid (T4).
- **patchlib:** patchers upgrade the previously deployed text in place (`previous=` blocks).

### G: build, deploy, runtime

- **config.json:**
  - A corrupt file is copied aside and reset in place instead of crash-looping; saves fsync (G-01).
  - run.sh creates `{}`.
- **run.sh (G-02, G-05):**
  - It keeps the replaced container until the new one answers and passes `smoke-test.sh` (which now requires CUDA and
    the expected scripts).
  - It rolls back otherwise.
  - It stops gracefully.
- **Startup:**
  - Fails without CUDA unless the command line chooses CPU (G-03).
  - `PYTHONUNBUFFERED` (G-06).
  - wandb is not imported (G-09, about 0.6 s).
- **Provenance (G-07).**
  - OCI labels: revision, version, base digest.
  - The build uses the base image by digest.
  - run.sh reads the version from the labels.
- **Build hygiene:**
  - Protected-package pins are validated (G-10).
  - The patched site-packages are recompiled (G-11).
- **Caches (G-12).** `Caches/app` is mounted at `cache/`, with `TORCH_HOME` and `HF_HOME` inside it.
- **Mirror (G-13).** The extension mirror has no `--checksum`.
- **Allocator (GPU A/B).** run.sh sets `PYTORCH_ALLOC_CONF=expandable_segments:True`.
  - Output is pixel-identical and speed the same.
  - The reserved peak drops from 22.57 to 19.24 GB, host RAM on unified memory.
  - This resolves the open item from the correctness audit.

## Contract changes

- **Requests that now fail with an error naming the cause:**
  - a LoRA that cannot be merged;
  - an unloadable hypernetwork;
  - `dyn < 1`;
  - I/F-mode images, bad ICC profiles, empty `init_images`;
  - a multi-sampler `snapshot_dir` outside its root;
  - SEG under Tiled Diffusion and its other unsupported layouts;
  - PAG with a split `to_v`;
  - converter overflows and an unknown `bake_in_vae`;
  - SDXL region prompt control.
- **New infotext entries:** `Lora errors` (unmatched LoRA keys); s_* values when they differ from the sampler
  defaults; `Hires schedule type: Automatic`.
- **Startup:**
  - CPU-only starts need `--skip-torch-cuda-test` (or `--use-cpu all`).
  - run.sh refuses an image without provenance labels (deploy10 and older) unless `A1111_COMMIT_HASH` and
    `A1111_VERSION_TAG` are given.

## Measured, not adopted

| Item | Reason |
|---|---|
| `allow_bf16_reduced_precision_reduction=(False, True)` (G-08) | Pixel-identical on both workloads and the same speed, so no split-K reduced-precision algorithm is selected for these shapes |
| No job-end `empty_cache` | Within noise |
| Parallel PNG writer for API responses and snapshot re-encodes (D-06, D-09) | Slower than Pillow on the 1280² production output |
| Deduplicated alternation encodes (A-18) | CLIP rows are not batch-size invariant, so not exact |
| PAG computing only V | Not exact under Hypertile's per-call RNG, and about 0.1% of a request |
| ControlNet hint identity fast path (M5, ControlNet side) | Inference tensors have no version counter, so in-place edits by other extensions could not be detected |
| ControlNet skip-concat prealloc, PAG skip-sum replay | Below the integrated measurement resolution (±0.03 s) |
| Eager LoRA merge so graphs survive LoRA-set changes | Only helps when the LoRA set changes; production reuses one set |
| Integer even tile spacing (M1) | Differs from upstream in 14,198 configs that upstream covers correctly |
| Hypertile SD-XL depth table (H2) | Upstream behaviour with no stated intent; documented |
| SEG as in the published pipeline (F-04) | A design option, not a defect |

## Owner items

- **System-Statistics.** `~/System-Statistics` runs run.sh with a 120 s subprocess timeout (SIGKILL). The new run.sh
  waits for readiness and the smoke test, so it can take longer. Raise that timeout (for example to 1200 s) before
  using the recreate or rebuild buttons.
- **API exposure (G-14).** The API listens on the LAN with no `--api-auth`, and `--enable-insecure-extension-access`
  is a dead flag.
- **Multi-sampler infotext.** The chain's eta never reaches the infotext.
- **img2img call count.** img2img makes one sampler call fewer than requested; this is the ldm reference behaviour
  (A-12 note).
- **Earlier deferrals** carried over from the cleanup pass: 3-32 and 3-62.

## VAE-only layout

The owner decided the parked NHWC GroupNorm / all-NCHW work stays parked, and deferred the VAE-only layout
re-evaluation (about 1.1 s of strided copies per n-w1 request at the VAE's full-resolution sizes, per the profile above)
to a later date. Nothing in this pass changes the layout.
