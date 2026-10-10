# STATUS

## Mission

Run AUTOMATIC1111 as a GB10-native, API-only appliance on the NVIDIA NGC PyTorch base. That means:

- keep the repo reviewable
- keep the NGC-tuned framework stack protected
- keep the host-mounted user-data layout intact

## Production (checked 2026-10-10)

- **Running image.** `gb10-a1111-latest` runs `local/gb10-a1111:deploy11-739bc58b`, which is also `latest` (image ID
  `sha256:66ba864d7c5d...`, labelled with its revision, version and base digest). Deployed 2026-10-10 from commit
  `739bc58b`, it adds the [2026-10-09 correctness, quality and speed pass](notes/correctness-quality-speed-pass-2026-10-09.md)
  to deploy10's [cleanup pass](notes/cleanup-pass-2026-10-09.md).
- **Deploy.** It was the first one through the health-gated run.sh. The smoke test passed (CUDA, the 10 expected
  scripts, the precision map with fp32 text encoders), and deploy10 was removed only afterwards. The deploy log is
  under `deploy-logs/`.
- **Live verification** (neutral prompt, the operator's last img2img request):
  - 11.26 s, the same pixels as the validated candidate (`be72db5a687e8956`).
  - The candidate's live API tests passed 35/35.
  - Its UNet/VAE graphs were bit-identical to eager.
  - Images differ from deploy10 by design: PAG `to_out`, the CLIP-L LoRA keys, fp32 text encoders and VAE input
    rounding. See the note.
- **Builds.** `gb10/build.sh` and `gb10/run.sh` from this checkout (branch `latest`). System-Statistics' Rebuild button
  runs the same two scripts, and its Recreate timeout covers run.sh's health gate (2400 s, System-Statistics
  `134578f`).

### Images kept for rollback

Roll back with `IMAGE_TAG=local/gb10-a1111:<tag> gb10/run.sh`. Images before deploy11 predate the provenance labels, so
for them also set `A1111_COMMIT_HASH` and `A1111_VERSION_TAG` to the image's commit and that commit's
`git describe --tags`
([launch/README.md](launch/README.md#deploy-gb10runsh)).

| Tag | Image ID | Contents |
|---|---|---|
| `deploy11-739bc58b` (= `latest`) | `66ba864d7c5d` | production: the 2026-10-09 correctness, quality and speed pass |
| `deploy10-c22a9794` | `c4dcc49691e9` | the 2026-10-09 cleanup pass on top of deploy9 |
| `deploy9-75a94f59` | `2708c45d4d9d` | correctness audit + performance pass 2 |
| `deploy8-2b5e4039` | `e4501073fe84` | correctness audit plus its follow-ups (image-URL byte/pixel budgets, atomic patchers, sampler-registry publication) |
| `deploy7-490eac83` | `db2612552043` | correctness audit, first deploy (live-verified in the audit note) |
| `deploy5-293d3e2c` | `59626c7c24bc` | [2026-10-06 static performance pass](notes/performance-static-pass-2026-10-06.md) |
| `pre-perf-20261006` | `7f954f2fc0fc` | the 2026-09-27 deploy4 build, before the performance passes |

## Current defaults

- **Base image.** `nvcr.io/nvidia/pytorch:26.08-py3`, built by digest (`sha256:3becd068f49b...`). MSLK is built from source at `88d06bc`, and torch extensions
  target `12.1a` (details in [README.md](README.md)).
- **Protected packages.** Every NGC package is protected except the 50 stock PyPI wheels in
  `docker/base-released-packages.txt`.
- **Launch flags.** The launcher's API-only defaults are used, including `--opt-channelslast`, `--dtype bfloat16` and
  `--precision autocast` ([launch/README.md](launch/README.md#launch-flags)).
- **run.sh runtime defaults.**
  - SDPA backend order `cudnn,flash,efficient,math`
  - UNet CUDA graphs on (cache 8)
  - VAE decode CUDA graphs on (cache 4)
  - compile caches under `/opt/gb10/stable-diffusion/Caches/compile`, namespaced by the image stack and the driver
  - app caches and torch.hub/Hugging Face downloads under `/opt/gb10/stable-diffusion/Caches/app`
  - `PYTORCH_ALLOC_CONF=expandable_segments:True`
- **Text encoders** run in fp32 (autocast off, IEEE matmul); the UNet and VAE stay bf16.
- **NHWC GroupNorm kernels** are off (`/sdapi/v1/openclaw/nhwc-groupnorm`). TeaCache is off by default.
- **Production settings, read via `GET /sdapi/v1/options` on 2026-10-09:**
  - checkpoint `MM_R2_FIX`, VAE `ftasticVAE_v10.safetensors`
  - `mxfp8_storage` and `nvfp4_storage` both `Disable`
  - cross-attention `sdp - scaled dot product`
- **Build CPU placement.** Builds run in `gb10build.slice` (performance cores 5-9,15-19) once those CPUs are no longer
  boot-isolated with `isolcpus=domain`. Until then they use the default efficiency-core placement.

## Persistent host surfaces

The host root is `/opt/gb10/stable-diffusion`. [launch/README.md](launch/README.md#host-mounts) has the full mount
table. The host-owned surfaces are:

- `config/` (`config.json`, `styles.csv`, `generation-last/`)
- `Models/` (checkpoints)
- the model directories `BLIP`, `CLIP`, `Codeformer`, `GFPGAN`, `karlo`, `RealESGRAN`, `torch_deepdanbooru`, `VAE` and
  `VAE-approx`
- `Extensions/`
- `Caches/app/` (the app's `cache/`, with torch.hub and Hugging Face downloads) and `Caches/compile/`
- `Embeddings/`, `Hypernetworks/`, `Lora/` and `Outputs/`, which are symlinks into `/mnt/nas-warehouse/StableDiffusion/`

The host root also has directories that run.sh does not mount. They are leftovers from earlier layouts, and nothing in
the container reads them:

- `BSRGAN`, `cache`, `Cache`, `ControlNet`, `deepbooru`, `ExtensionPatchBackups`, `Extensions.quarantine`, `LDSR`
- `RealESRGAN`, `Repositories`, `runtime-notes`, `ScuNET`, `Stable-diffusion`, `SwinIR`

## Owned extension posture

- Seven extensions are owned source under `extensions/`: ControlNet, Incantations, TeaCache, the model converter and
  three `openclaw-*` helpers. They carry upstream provenance and license copies.
- `gb10/run.sh` mirrors them into the host `Extensions/` on every deploy. The image does not contain them.
- Fixes to PAG/SEG/CFG-combiner/Dynamic Thresholding, ControlNet and TeaCache are made here. They are never made in an
  untracked host checkout.
- Three third-party extensions stay host-installed and deploy-patched: MultiDiffusion, Ultimate Upscale and
  detail-daemon. See [EXTENSIONS.md](EXTENSIONS.md).

## Parked work

- **NHWC GroupNorm Triton kernels** were 21-30% faster per request. The code and the runtime switch are kept, with the
  switch off.
- **All-NCHW layout** (no `--opt-channelslast`) was 8.5-16% faster.
- Both hard-locked the host under sustained load on 2026-10-07. Keep `--opt-channelslast` on and the NHWC switch off
  until the host is stable under sustained GPU load. The retry protocol is under "Re-evaluating the layout work" in
  [performance pass 2](notes/performance-pass-2-2026-10-07.md).

## Open items

1. The owner items the cleanup pass deferred are listed in [notes/cleanup-pass-2026-10-09.md](notes/cleanup-pass-2026-10-09.md).
2. Owner decisions on host leftovers: the directories listed above that run.sh does not mount, and the quarantine copy
   `Extensions.quarantine/20260503-194044/sd-webui-prompt-all-in-one`.
3. Adopt or replace the third-party extensions, starting with MultiDiffusion and detail-daemon (EXTENSIONS.md).
4. The VAE-only layout re-evaluation (NHWC GroupNorm or an NCHW VAE), deferred by the owner on 2026-10-09; see
   [notes/correctness-quality-speed-pass-2026-10-09.md](notes/correctness-quality-speed-pass-2026-10-09.md).

The API has no authentication by design: it is a local container on a trusted LAN.

History before 2026-10 lives in git and in [notes/](notes/README.md).
