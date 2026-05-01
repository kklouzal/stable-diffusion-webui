#!/usr/bin/env python3
import importlib.metadata as md
import json
import os
import re
import subprocess
from pathlib import Path

BASE_CONSTRAINTS = Path(os.environ.get('BASE_CONSTRAINTS', '/opt/base-python-protected-constraints.txt'))
DIRECT_REQUIREMENTS = Path(os.environ.get('DIRECT_REQUIREMENTS', '/opt/requirements-image.txt'))
A1111_DIR = Path(os.environ.get('A1111_DIR', '/opt/stable-diffusion-webui'))
OUTPUT_TEXT = Path(os.environ.get('OUTPUT_TEXT', str(A1111_DIR / 'BUILD_MANIFEST.txt')))
OUTPUT_JSON = Path(os.environ.get('OUTPUT_JSON', str(A1111_DIR / 'BUILD_MANIFEST.json')))
PYTORCH_NIGHTLY_INDEX_URL = os.environ.get('PYTORCH_NIGHTLY_INDEX_URL', 'https://download.pytorch.org/whl/nightly/cu130')
PYTORCH_NIGHTLY_PKGS = {'torch', 'torchvision', 'torchaudio'}
EXTRA_DIRECT = {'clip'}


def normalize(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name.strip().lower())


def canonical_req_line(line: str) -> str:
    return re.sub(r'\s+', '', line.split('#', 1)[0].strip().lower())


def parse_req_name(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('-'):
        return None
    name = re.split(r'[<>=!~ ;\[]', line, 1)[0].strip()
    return normalize(name) if name else None


def load_constraint_map(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw or '==' not in raw:
            continue
        name, version = raw.split('==', 1)
        data[normalize(name)] = version.strip()
    return data


def load_req_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        name = parse_req_name(raw)
        if name and name not in out:
            out[name] = raw.strip()
    return out


base_pkgs = load_constraint_map(BASE_CONSTRAINTS)
repo_direct_map = load_req_map(DIRECT_REQUIREMENTS)
upstream_versions_map = load_req_map(A1111_DIR / 'requirements_versions.txt')
upstream_plain_map = load_req_map(A1111_DIR / 'requirements.txt')
upstream_direct = (set(upstream_versions_map) | set(upstream_plain_map)) - {'torch'}
repo_direct = set(repo_direct_map) | EXTRA_DIRECT

all_dists: dict[str, dict] = {}
for dist in md.distributions():
    name = dist.metadata.get('Name')
    if not name:
        continue
    norm = normalize(name)
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


latest_cache: dict[tuple[str, str | None], str] = {}


def latest_visible(name: str, extra_index_url: str | None = None) -> str:
    key = (name, extra_index_url)
    if key in latest_cache:
        return latest_cache[key]
    if name == 'clip':
        latest_cache[key] = 'source-archive'
        return latest_cache[key]
    cmd = ['python', '-m', 'pip', 'index', 'versions', '--disable-pip-version-check']
    if extra_index_url:
        cmd.extend(['--extra-index-url', extra_index_url])
    cmd.append(name)
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        latest_cache[key] = 'query-failed'
        return latest_cache[key]
    result = 'unknown'
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('Available versions:'):
            vals = [x.strip() for x in line.split(':', 1)[1].split(',') if x.strip()]
            result = vals[0] if vals else 'unknown'
            break
    latest_cache[key] = result
    return result


def direct_reason(name: str) -> str:
    hoisted = name in base_pkgs
    if name == 'clip':
        return 'Repo-Built-Wheel|Hoisted-Into-Base' if hoisted else 'Repo-Built-Wheel'
    repo_line = repo_direct_map.get(name, '')
    upstream_line = upstream_versions_map.get(name, '')
    if upstream_line and canonical_req_line(repo_line) == canonical_req_line(upstream_line):
        return 'A1111-Upstream-Pin|Hoisted-Into-Base' if hoisted else 'A1111-Upstream-Pin'
    if name in upstream_direct:
        return 'Repo-Selected-Direct|A1111-Override|Hoisted-Into-Base' if hoisted else 'Repo-Selected-Direct|A1111-Override'
    return 'Repo-Selected-Direct|Hoisted-Into-Base' if hoisted else 'Repo-Selected-Direct'


def base_reason(name: str) -> str:
    if name in PYTORCH_NIGHTLY_PKGS:
        return 'Base-Provided|PyTorch-Nightly'
    if name.startswith('nvidia-') or name.startswith('cuda-') or name == 'triton':
        return 'Base-Provided|Torch-CUDA-Stack'
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
        item['source_reason'] = direct_reason(name)
        item['repo_direct_entry'] = repo_direct_map.get(name)
        item['upstream_versions_entry'] = upstream_versions_map.get(name)
        item['upstream_requirements_entry'] = upstream_plain_map.get(name)
        sections['direct'].append(item)
    elif name in base_pkgs:
        item['category'] = 'base'
        item['source_reason'] = base_reason(name)
        sections['base'].append(item)
    else:
        item['category'] = 'indirect'
        item['source_reason'] = indirect_reason(name)
        sections['indirect'].append(item)

for items in sections.values():
    for item in items:
        extra = PYTORCH_NIGHTLY_INDEX_URL if item['normalized'] in PYTORCH_NIGHTLY_PKGS else None
        item['latest'] = latest_visible(item['normalized'], extra)

lines: list[str] = []
lines.append('=== GB10 A1111 build manifest ===')
lines.append('')
lines.append('[classification summary]')
lines.append('base-layer-provided = framework/base packages present before the A1111 direct-dependency hoist')
lines.append('a1111-direct = explicitly selected by upstream A1111 requirements or repo-owned direct additions, whether hoisted into base or not')
lines.append('a1111-indirect = transitive dependencies pulled in under the direct set')
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
        line = f"{item['name']} ({item['installed']})"
        latest = item['latest']
        if latest != item['installed']:
            line += f" --> Latest: {latest} [{item['source_reason']}]"
        else:
            line += f" [{item['source_reason']}]"
        lines.append(line)
    lines.append('')
text = '\n'.join(lines).rstrip() + '\n'
OUTPUT_TEXT.write_text(text)
OUTPUT_JSON.write_text(json.dumps({
    'summary': {k: len(v) for k, v in sections.items()},
    'pytorch_nightly_index_url': PYTORCH_NIGHTLY_INDEX_URL,
    'upstream_direct_count': len(upstream_direct),
    'repo_direct_count': len(repo_direct),
    'packages': sections,
}, indent=2) + '\n')
print(text)
