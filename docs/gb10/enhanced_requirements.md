# enhanced_requirements.md

This file is the **controlled widening ledger** for the GB10-A1111 container.

Use it to record package-by-package or cluster-by-cluster experiments that try newer versions than the current proven baseline in `baseline_requirements.md`.

## Purpose

This file is **not** the baseline.
It is the experimental record of what we tried to widen, what passed, what failed, and what became safe to carry forward.

If a widening survives real validation, record it here before promoting it into the repo baseline.

---

## Validation bar for a widening to count as “passed”

A widening is not considered successful just because the image builds.

Minimum bar:

- image builds successfully
- container starts successfully
- A1111 serves UI/API successfully
- model loads successfully
- at least one real image generation succeeds
- no framework ownership regression occurred
  - `torch` still from NVIDIA base image
  - `torchvision` still from NVIDIA base image
  - CUDA posture still intact

Recommended extra checks when relevant:

- benchmark against the current baseline
- check for new warnings/errors in logs
- validate that extensions / LoRA / embeddings / host-side mounted/symlinked asset visibility did not regress

---

## Widening workflow template

For each attempt, record:

- date
- branch / commit if relevant
- package or package cluster widened
- previous version
- attempted version
- reason for trying it
- build result
- runtime result
- image generation result
- notes / regressions / conclusions
- whether it should be promoted, parked, or rejected

---

## Passed widenings

### 2026-04-24 — `open-clip-torch` widening for current SDXL path

- **branch / commit context:** local worktree on `main` during the `open-clip-torch` compatibility lane
- **package widened:** `open-clip-torch`
- **previous version:** `2.24.0` in `docker/requirements-image.txt`
- **attempted version:** latest resolved current release, then pinned after validation as `3.3.0`
- **reason:** newer `open-clip-torch` changed transformer layout expectations (`batch_first`) and required explicit compatibility handling in the vendored `generative-models` OpenCLIP path
- **build result:** passed via `IMAGE_TAG=local/gb10-a1111:open-clip-multilora-probe ./scripts/build.sh`
- **runtime result:** passed; container started, API served on port `7862`, SDXL model `Mega555_00001_` loaded successfully
- **image generation result:** passed; one real SDXL txt2img image returned successfully with multiple LoRA tags in the prompt, and an additional 3-LoRA API pass also completed successfully
- **notes / regressions / conclusions:** required local patch `patches/generative-models/0001-open-clip-batch-first-compat.patch` so newer OpenCLIP transformer blocks work whether attention is sequence-first or batch-first. Startup emitted a one-time `ViT-bigG-14` "No pretrained weights loaded" warning before checkpoint application, but real model load and generation still succeeded with no framework ownership drift (`torch 2.13.0.dev20260422+cu130`, `torchvision 0.27.0.dev20260423+cu130`, `open-clip-torch 3.3.0`).
- **decision:** promote / keep

### 2026-04-24 — `blendmodes` widening for numpy-2 current stack

- **branch / commit context:** local worktree on `main` immediately after the `open-clip-torch` widening landed
- **package widened:** `blendmodes`
- **previous version:** `2024.1.1` in the resolved direct package set
- **attempted version:** latest visible release, then pinned after validation as `2025`
- **reason:** continue the controlled direct-dependency widening lane with the smallest remaining top-level target first
- **build result:** passed via `IMAGE_TAG=local/gb10-a1111:blendmodes-probe ./scripts/build.sh` after adding a local resolver compatibility step for Gradio 3.50.2
- **runtime result:** passed; container started, API served on port `7862`, SDXL model `Mega555_00001_` loaded successfully
- **image generation result:** passed; a real SDXL txt2img image returned successfully with multiple LoRA tags in the prompt
- **notes / regressions / conclusions:** first resolver attempt failed because `blendmodes 2025` requires `numpy>=2.0.2` while the Gradio wheel metadata still advertises `numpy~=1.0`. The repo therefore prepares a resolver-only patched Gradio wheel that relaxes that stale metadata to `numpy>=1.0` before generating the dry-run dependency report. `blendmodes 2025` itself remained good, but the later experimental runtime widening to Gradio 3.50.2 turned out to be UI-hostile on this stack and was rolled back to the upstream-style `gradio==3.41.2` pin. No framework ownership drift occurred (`torch 2.13.0.dev20260422+cu130`, `torchvision 0.27.0.dev20260423+cu130`, `blendmodes 2025`).
- **decision:** promote / keep

---

## Failed or rejected widenings

### 2026-04-24 — `gradio` runtime widening rejected; rolled back to upstream pin

- **branch / commit context:** local worktree on `main` after the successful `blendmodes` widening
- **package widened:** `gradio`
- **previous version:** `3.41.2` (upstream-style baseline)
- **attempted version:** `3.50.2`
- **reason:** test the next remaining top-level UI/framework lane after `blendmodes`
- **build result:** build/install path succeeded
- **runtime result:** rejected; the live WebUI surfaced broken loading behavior, malformed control metadata, and follow-on UI incompatibilities that required temporary in-container hotfixes just to recover basic rendering
- **image generation result:** not accepted as sufficient evidence to keep the widening, because the user-facing surface was materially broken
- **notes / regressions / conclusions:** A1111 on this stack is much happier on the upstream-style `gradio==3.41.2` line than on `3.50.2`. Recreating the container from the rebuilt `3.41.2` image discards the temporary live hotfixes and restores a clean UI shape (`undefined` probe returned zero matches, startup callbacks stayed green, and a real txt2img API call saved `/mnt/server-002-models/StableDiffusion/Outputs/txt2img-images/2026-04-25/00000-12345.png`). Keep `gradio` out of the casual widening lane unless there is a strong reason to re-open it as a dedicated compatibility project.
- **decision:** reject / revert to `3.41.2`

---

## Deferred candidates

These are plausible future widening targets, but not yet approved as safe:

- `tokenizers`
- `torchmetrics`
- `lightning-utilities`
- other supplemental compatibility packages that are not simply inherited from A1111’s current `requirements_versions.txt`

Core upstream-pinned packages should **not** be widened casually. If upstream already pins an exact version in the baked A1111 tree, treat widening that package as an explicit experiment rather than a presumed upgrade.

---

## Notes

- Preserve the current working baseline in `baseline_requirements.md`.
- Prefer one-package or one-cluster experiments over broad churn.
- If a package widening requires widening multiple tightly coupled packages, record that as a cluster experiment.
- If a package widening forces framework ownership drift or starts fighting the NVIDIA base image, reject it unless there is a very strong reason to redesign the stack.
