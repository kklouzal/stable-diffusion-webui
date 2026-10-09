#!/usr/bin/env python3
"""Write BUILD_MANIFEST.txt/.json: every installed Python distribution classified as base (protected NGC/torch
stack), direct (repo-owned requirement) or indirect (pulled in by a direct one), with its installed version."""
import importlib.metadata as md
import json
import os
import re
from pathlib import Path

BASE_CONSTRAINTS = Path('/opt/build/base-python-protected-constraints.txt')
RELEASED_FLOORS = Path('/opt/build/base-python-released-floors.txt')
A1111_DIR = Path('/opt/stable-diffusion-webui')
# Repo-owned direct requirements: the app closure plus the image's ControlNet supplement.
DIRECT_REQUIREMENTS = (
    A1111_DIR / 'requirements_versions.txt',
    Path('/opt/build/requirements-sd-webui-controlnet-image.txt'),
)
OUTPUT_TEXT = A1111_DIR / 'BUILD_MANIFEST.txt'
OUTPUT_JSON = A1111_DIR / 'BUILD_MANIFEST.json'
NGC_PYTORCH_PKGS = {'torch', 'torchvision', 'torchaudio'}
OPTIONAL_ABSENT = {'torchaudio'}
TORCH_QUANTIZATION_PKGS = {'torchao', 'mslk'}
MSLK_SOURCE_COMMIT = os.environ.get('MSLK_SOURCE_COMMIT')
EXTRA_DIRECT = {'clip'}


def normalize(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name.strip().lower())


def parse_req_name(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('-'):
        return None
    name = re.split(r'[<>=!~ ;\[]', line, maxsplit=1)[0].strip()
    return normalize(name) if name else None


def load_constraint_map(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw or '==' not in raw:
            continue
        name, version = raw.split('==', 1)
        data[normalize(name)] = version.strip()
    return data


def load_req_map(paths) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        for raw in path.read_text().splitlines():
            name = parse_req_name(raw)
            if name and name not in out:
                out[name] = raw.strip()
    return out


def load_floor_map(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if raw and not raw.startswith('#') and '>=' in raw:
            name, version = raw.split('>=', 1)
            data[normalize(name)] = version.strip()
    return data


base_pkgs = load_constraint_map(BASE_CONSTRAINTS)
released_floors = load_floor_map(RELEASED_FLOORS)
repo_direct_map = load_req_map(DIRECT_REQUIREMENTS)
repo_direct = set(repo_direct_map) | EXTRA_DIRECT

all_dists: dict[str, dict] = {}
for dist in md.distributions():
    name = dist.metadata.get('Name')
    if not name:
        continue
    norm = normalize(name)
    if norm in all_dists:
        continue  # shadowed copy later on sys.path; the first one is what Python imports
    all_dists[norm] = {
        'display': name,
        'version': dist.version,
        'requires': dist.requires or [],
    }

explicit_direct = repo_direct & set(all_dists)

reverse = {name: set() for name in all_dists}
for parent, info in all_dists.items():
    for req in info['requires']:
        dep = parse_req_name(req)
        if dep and dep in all_dists:
            reverse[dep].add(parent)

root_cache: dict[str, list[str]] = {}


def roots_for(pkg: str) -> list[str]:
    cached = root_cache.get(pkg)
    if cached is not None:
        return cached
    seen = set()
    roots = set()
    stack = [pkg]
    while stack:
        cur = stack.pop()
        for parent in reverse.get(cur, ()):
            if parent in seen:
                continue
            seen.add(parent)
            if parent in explicit_direct:
                roots.add(parent)
            else:
                stack.append(parent)
    out = sorted(roots)
    root_cache[pkg] = out
    return out


def released_tag(name: str) -> str:
    floor = released_floors.get(name)
    return f'|Released-From-NGC:{floor}' if floor else ''


def direct_reason(name: str) -> str:
    # Only called for explicit_direct names, i.e. 'clip' or a DIRECT_REQUIREMENTS entry.
    hoisted = name in base_pkgs
    if name == 'clip':
        return 'Repo-Built-Wheel|Hoisted-Into-Base' if hoisted else 'Repo-Built-Wheel'
    return 'Repo-Owned-Requirement|Hoisted-Into-Base' if hoisted else 'Repo-Owned-Requirement'


def base_reason(name: str) -> str:
    if name in OPTIONAL_ABSENT and name not in all_dists:
        return 'Base-Provided|NGC-PyTorch|Optional-Absent-Allowed'
    if name in NGC_PYTORCH_PKGS:
        return 'Base-Provided|NGC-PyTorch'
    if name.startswith('nvidia-') or name.startswith('cuda-') or name == 'triton':
        return 'Base-Provided|Torch-CUDA-Stack'
    if name == 'mslk' and MSLK_SOURCE_COMMIT:
        return f'Base-Provided|Torch-Quantization-Stack|Source-Built:{MSLK_SOURCE_COMMIT[:12]}'
    if name in TORCH_QUANTIZATION_PKGS:
        return 'Base-Provided|Torch-Quantization-Stack'
    return 'Base-Provided|Torch-Base'


def indirect_reason(name: str) -> str:
    roots = roots_for(name)
    if roots:
        shown = ', '.join(roots[:4])
        if len(roots) > 4:
            shown += ', ...'
        return f'Indirect via {shown}'
    return 'Indirect'


sections = {'base': [], 'direct': [], 'indirect': []}
for name in sorted(all_dists):
    info = all_dists[name]
    item = {
        'name': info['display'],
        'normalized': name,
        'installed': info['version'],
        'roots': roots_for(name),
    }
    if name in explicit_direct:
        item['category'] = 'direct'
        item['source_reason'] = direct_reason(name) + released_tag(name)
        item['repo_direct_entry'] = repo_direct_map.get(name)
        sections['direct'].append(item)
    elif name in base_pkgs:
        item['category'] = 'base'
        item['source_reason'] = base_reason(name)
        sections['base'].append(item)
    else:
        item['category'] = 'indirect'
        item['source_reason'] = indirect_reason(name) + released_tag(name)
        sections['indirect'].append(item)

for name in sorted(OPTIONAL_ABSENT):
    if name not in all_dists:
        sections['base'].append({
            'name': name,
            'normalized': name,
            'installed': 'optional-absent',
            'roots': [],
            'category': 'base',
            'source_reason': base_reason(name),
            'optional_absent': True,
        })

lines: list[str] = []
lines.append('=== GB10 A1111 build manifest ===')
lines.append('')
lines.append('[classification summary]')
lines.append('base-layer-provided = CUDA/PyTorch/base packages protected before A1111 app dependency installation')
lines.append('a1111-direct = explicitly selected by repo-owned requirements_versions.txt or the ControlNet image supplement; base matches stay protected')
lines.append('a1111-indirect = transitive dependencies pulled in under the direct set')
lines.append('Released-From-NGC:<version> = NGC stock wheel released to the app resolver (docker/base-released-packages.txt); <version> is the NGC floor')
lines.append('torchaudio = optional for the NGC CUDA 13.4 lane; absence is accepted unless a runtime import requirement is proven')
lines.append(f"base_layer_provided: {len(sections['base'])}")
lines.append(f"a1111_direct: {len(sections['direct'])}")
lines.append(f"a1111_indirect: {len(sections['indirect'])}")
lines.append('')
for key, title in (
    ('base', '[base-layer-provided python packages]'),
    ('direct', '[a1111 direct python packages]'),
    ('indirect', '[a1111 indirect python packages]'),
):
    lines.append(title)
    for item in sections[key]:
        lines.append(f"{item['name']} ({item['installed']}) [{item['source_reason']}]")
    lines.append('')
text = '\n'.join(lines).rstrip() + '\n'
OUTPUT_TEXT.write_text(text)
OUTPUT_JSON.write_text(json.dumps({
    'summary': {k: len(v) for k, v in sections.items()},
    'released_from_ngc': released_floors,
    'repo_direct_count': len(repo_direct),
    'packages': sections,
}, indent=2) + '\n')
print(text)
