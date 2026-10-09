# GB10 companion-repository patches

This directory holds source patches for upstream companion repositories that are
still cloned during the GB10 Docker image build.

The `stable-diffusion-webui` patches formerly carried by GB10-A1111 have been
applied directly to this fork's `latest` branch and are intentionally not kept
here as build-time patches.

Build-applied patch targets:

- `patches/stable-diffusion-stability-ai/*.patch`
- `patches/generative-models/*.patch`

These are applied in lexical order within each target directory by `docker/apply-local-patches.py`.

Host-mounted third-party extensions (MultiDiffusion, Ultimate Upscale) are not patched here: `gb10/run.sh` patches
them in place at deploy time with the `gb10/patch-*.py` patchers, which accept pristine upstream checkouts.
