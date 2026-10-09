#!/usr/bin/env python3
"""Turn the resolver dry-run report into name==version pins for the wheel build, failing the build if the
resolved application closure contains any package the NVIDIA base owns."""
import json
import re
from pathlib import Path

REPORT = Path('/opt/build/report.json')
TARGET = Path('/opt/build/requirements-resolved.txt')
PROTECTED_NAMES = {'torch', 'torchvision', 'torchaudio', 'triton'}
PROTECTED_PREFIXES = ('nvidia-', 'cuda-')
PROTECTED_NAMES_FILE = Path('/opt/build/base-python-protected-names.txt')


def normalize(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name.strip().lower())


base_protected_names = {
    normalize(line)
    for line in PROTECTED_NAMES_FILE.read_text().splitlines()
    if line.strip() and not line.lstrip().startswith('#')
}


def protected(norm: str) -> bool:
    return norm in PROTECTED_NAMES or norm in base_protected_names or norm.startswith(PROTECTED_PREFIXES)


report = json.loads(REPORT.read_text())
reqs = []
blocked = []
for item in report.get('install', []):
    meta = item.get('metadata', {})
    name = meta.get('name')
    version = meta.get('version')
    if name and version:
        norm = normalize(name)
        if protected(norm):
            blocked.append(f'{norm}=={version}')
            continue
        reqs.append(f'{name}=={version}')

if blocked:
    raise SystemExit(f'protected NVIDIA base packages resolved as app deps: {", ".join(blocked)}')

TARGET.write_text('\n'.join(reqs) + '\n')
print(f'resolved {len(reqs)} application packages into {TARGET}')
