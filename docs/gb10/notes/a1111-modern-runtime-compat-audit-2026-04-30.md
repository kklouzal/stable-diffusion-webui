# A1111 modern runtime compatibility audit — 2026-04-30

Scope: running GB10 A1111 image source under `/opt/stable-diffusion-webui`, bundled repositories, builtin extensions, and mounted extensions.

Runtime package baseline observed before this patch set:

- Python 3.12.3
- torch 2.13.0.dev20260430+cu132
- pytorch_lightning 2.6.1
- transformers 5.7.0
- gradio 3.41.2
- open_clip 3.3.0

## High-confidence patch candidates applied

### PyTorch Lightning 2.x compatibility

A1111 printed `Pytorch_lightning.distributed not found, attempting pytorch_lightning.rank_zero` because Lightning 2.x removed the old `pytorch_lightning.utilities.distributed` module path. The modern import is `pytorch_lightning.utilities.rank_zero`.

Patched:

- `modules/initialize_util.py` keeps a silent legacy alias for third-party code that still imports the old module name.
- `modules/models/diffusion/ddpm_edit.py` imports `rank_zero_only` from the modern path.
- `extensions-builtin/LDSR/sd_hijack_ddpm_v1.py` imports `rank_zero_only` from the modern path.
- `repositories/stable-diffusion-stability-ai/ldm/models/diffusion/ddpm.py` imports `rank_zero_only` from the modern path.

This removes the startup warning and makes the local source use the current Lightning API where the source itself had hard-coded old imports.

### Modern PyTorch AMP API

PyTorch nightly emits `FutureWarning` for `torch.cuda.amp.autocast(...)`, `torch.cpu.amp.autocast(...)`, and `torch.cuda.amp.GradScaler(...)`. The current API is `torch.amp.autocast(device_type, ...)` and `torch.amp.GradScaler(device, ...)`.

Patched high-confidence local uses:

- A1111 training paths:
  - `modules/hypernetworks/hypernetwork.py`
  - `modules/textual_inversion/textual_inversion.py`
- Stable Diffusion dependency repo:
  - `ldm/util.py`
  - `ldm/modules/diffusionmodules/util.py`
  - `ldm/modules/karlo/kakao/sampler.py`
  - `scripts/txt2img.py`
- Generative Models dependency repo:
  - `sgm/util.py`
  - `sgm/modules/diffusionmodules/util.py`
  - `main.py`

### Python 3.12 / NumPy 2 cleanup

Patched:

- `repositories/stable-diffusion-stability-ai/ldm/models/diffusion/dpm_solver/dpm_solver.py` raw docstring to remove a Python 3.12 invalid-escape `SyntaxWarning`.
- `repositories/stable-diffusion-stability-ai/ldm/modules/image_degradation/utils_image.py` replaces removed `np.int` with builtin `int`.
- `modules/textual_inversion/autocrop.py` replaces deprecated `pkg_resources.parse_version` import with `packaging.version.parse`.

### Extension dependency noise

Added `send2trash` to the image requirements so the mounted `sd_delete_button` extension can use its intended recycle-bin path instead of logging:

`Delete Button: send2trash is not installed. recycle bin cannot be used.`

## Items inspected but not patched yet

### xformers

Current logs include `no module xformers. Processing without...`. This is expected under the current GB10/CUDA 13/aarch64 stack. A1111 is already launched with `--opt-sdp-attention`, and the local source has a SageAttention backend patch. Installing or forcing xformers here is not a safe cleanup patch; it is a separate compatibility/build lane.

### HF_TOKEN warning

`Warning: You are sending unauthenticated requests to the HF Hub...` is a rate-limit/auth warning, not a source API compatibility issue. For this GB10 baseline it is accepted noise; do not inject an HF token just to silence it.

### Gradio 3.41.2

Gradio is intentionally kept pinned because A1111 is sensitive to Gradio API changes. Existing `gr.update(...)` usage is normal for this pin and should not be mass-changed.

### Mounted extension source

Mounted extensions were scanned. Notable findings:

- `sd-webui-prompt-all-in-one` has one Python 3.12 invalid-escape warning in `scripts/physton_prompt/translators/server.py` and two `distutils.version.LooseVersion` imports. These are extension-repo changes, not Docker source patches, because mounted extensions come from `/opt/gb10/stable-diffusion/Extensions` at runtime.
- `multidiffusion-upscaler-for-automatic1111` has optional xformers integration, but it also supports non-xformers attention modes and should not be forced toward xformers in this environment.
- `sd-webui-model-converter` uses `torch.load`; that is normal for local conversion behavior and not an API modernization issue by itself.

## Verification performed before rebuild

- Fresh A1111 `dev` checkout + all `patches/stable-diffusion-webui/*.patch` applied cleanly.
- Fresh `w-e-w/stablediffusion` checkout at `cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf` + `patches/stable-diffusion-stability-ai/*.patch` applied cleanly.
- Fresh `Stability-AI/generative-models` checkout at `45c443b316737a4ab6e40413d7794a7f5657c19f` + `patches/generative-models/*.patch` applied cleanly.
- Patched touched files compiled with `python3 -Werror::SyntaxWarning -m py_compile`.

## Second-pass performance/runtime hot-path audit

Additional high-confidence patches added after the broader A1111 + extension source pass:

- `modules/sd_hijack_optimizations.py` now uses the modern `torch.nn.attention.sdpa_kernel(...)` context with explicit `SDPBackend.FLASH_ATTENTION` and `SDPBackend.MATH` instead of deprecated `torch.backends.cuda.sdp_kernel(...)` for the `sdp-no-mem` attention paths.
- `modules/devices.py` now passes the selected A1111 dtype explicitly into CUDA autocast and uses `torch.is_autocast_enabled("cuda")`, matching the modern PyTorch device-scoped autocast state API.
- `modules/processing.py` now wraps the main image generation hot path in `torch.inference_mode()` instead of `torch.no_grad()`. This is deliberately scoped to generation, not training/checkpoint recomputation code.
- `modules/safe.py` now accepts PyTorch nightly zip metadata entries `.format_version` and `.storage_alignment`, then explicitly defaults accepted loads to `weights_only=False` after A1111's restricted-unpickle precheck. This preserves old checkpoint/embedding compatibility under modern PyTorch `torch.load` semantics without skipping A1111's safety gate.
- `repositories/generative-models` and `repositories/stable-diffusion-stability-ai` now use `torch.get_autocast_dtype("cuda")` and `torch.is_autocast_enabled("cuda")` where they snapshot CUDA autocast state.

### Mounted extension follow-up

`multidiffusion-upscaler-for-automatic1111/tile_utils/attn.py` was patched in the mounted extension checkout, not the Docker image patch stack, because runtime extensions are bind-mounted from `/opt/gb10/stable-diffusion/Extensions`. The local patch:

- recognizes A1111 `sage2` and `sage3` attention labels instead of warning/falling back to unoptimized vanilla attention for Tiled VAE AttnBlock code;
- maps those labels to SDP for that extension-specific VAE attention path because SageAttention's A1111 backend does not expose a compatible Tiled VAE AttnBlock implementation;
- replaces the extension's deprecated `torch.backends.cuda.sdp_kernel(...)` usage with `torch.nn.attention.sdpa_kernel(...)`;
- makes missing xformers explicit and falls back to SDP instead of relying on an undefined module name.

### Second-pass validation

- Rebuilt image: `local/gb10-a1111:runtime-hotpaths-check`.
- Runtime startup succeeded; `/internal/ping` responded.
- Attention API reported `sdpa`, `sage2`, and `sage3` available.
- `modules.safe` accepted a modern PyTorch zip checkpoint with `.format_version` / `.storage_alignment` metadata and loaded a state-dict payload successfully.
- `devices.autocast()` entered CUDA autocast with explicit dtype `torch.float16`.
- API txt2img smoke tests completed successfully under `sdpa`, `sage3`, `sage2`, and then `sdpa` again.
- Docker logs after the smoke test showed no traceback, source deprecation warning, `sdp_kernel` warning, or SageAttention failure; the only surfaced warning was the existing unauthenticated Hugging Face Hub rate-limit notice.

### Not patched after review

- Extension `torch.no_grad()` decorators were not blanket-replaced with `torch.inference_mode()`. Several are denoiser hijacks or tiled upscaler internals; the safe high-confidence inference-mode win is the central A1111 generation wrapper.
- `sd-webui-model-converter`'s `torch.load(..., map_location="cpu")` was not patched separately because A1111's global safe loader wrapper now handles modern PyTorch `weights_only` behavior after the restricted-unpickle precheck.
- Broad package upgrades were intentionally avoided. The current stack is already aggressive; source-level compatibility/performance fixes are lower-risk than changing Gradio/FastAPI/Pydantic/Transformers lanes again.
