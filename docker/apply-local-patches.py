#!/usr/bin/env python3
import subprocess
from pathlib import Path

BUILD_ROOT = Path('/opt/build')
PATCH_ROOT = Path('/opt/build/patches')
MOUNTED_EXTENSION_PATCH_ROOT = PATCH_ROOT / 'mounted-extensions'
TARGETS = {
    'stable-diffusion-webui': BUILD_ROOT / 'stable-diffusion-webui',
    'stable-diffusion-stability-ai': BUILD_ROOT / 'stable-diffusion-webui' / 'repositories' / 'stable-diffusion-stability-ai',
    'generative-models': BUILD_ROOT / 'stable-diffusion-webui' / 'repositories' / 'generative-models',
    'k-diffusion': BUILD_ROOT / 'stable-diffusion-webui' / 'repositories' / 'k-diffusion',
    'BLIP': BUILD_ROOT / 'stable-diffusion-webui' / 'repositories' / 'BLIP',
    'stable-diffusion-webui-assets': BUILD_ROOT / 'stable-diffusion-webui' / 'repositories' / 'stable-diffusion-webui-assets',
}


def run(cmd):
    print('+', ' '.join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    if not PATCH_ROOT.exists():
        print('no patches directory present; skipping')
        return

    applied = 0
    for patch_set, repo in TARGETS.items():
        patch_dir = PATCH_ROOT / patch_set
        if not patch_dir.exists():
            continue
        patches = sorted(p for p in patch_dir.iterdir() if p.is_file() and p.suffix == '.patch')
        if not patches:
            continue
        if not repo.exists():
            raise SystemExit(f'patch target missing for {patch_set}: {repo}')
        for patch in patches:
            run(['git', '-C', str(repo), 'apply', '--ignore-whitespace', '--check', str(patch)])
            run(['git', '-C', str(repo), 'apply', '--ignore-whitespace', str(patch)])
            applied += 1
            print(f'applied {patch.name} to {patch_set}', flush=True)

    if MOUNTED_EXTENSION_PATCH_ROOT.exists():
        print('mounted extension patches are present but not applied during image build; apply them to the corresponding host-mounted extension checkout when that extension is installed', flush=True)

    print(f'patch application complete; applied={applied}', flush=True)


if __name__ == '__main__':
    main()
