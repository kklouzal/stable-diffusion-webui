#!/usr/bin/env python3
import json
import re
from pathlib import Path

REPORT = Path('/opt/build/report.json')
TARGET = Path('/opt/build/requirements-resolved.txt')
AUDIT = Path('/opt/build/requirements-resolved-protected-audit.json')
PROTECTED_NAMES = {'torch', 'torchvision', 'torchaudio', 'triton'}
PROTECTED_PREFIXES = ('nvidia-', 'cuda-')
PROTECTED_NAMES_FILE = Path('/opt/build/base-python-protected-names.txt')


def normalize(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name.strip().lower())


base_protected_names = {
    normalize(line)
    for line in PROTECTED_NAMES_FILE.read_text().splitlines()
    if line.strip() and not line.lstrip().startswith('#')
} if PROTECTED_NAMES_FILE.exists() else set()


def protected(name: str) -> bool:
    norm = normalize(name)
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
            blocked.append({'name': name, 'normalized': norm, 'version': version})
            continue
        reqs.append(f'{name}=={version}')

AUDIT.write_text(json.dumps({
    'protected_names': sorted(PROTECTED_NAMES),
    'base_protected_names_file': str(PROTECTED_NAMES_FILE),
    'base_protected_names_count': len(base_protected_names),
    'protected_prefixes': list(PROTECTED_PREFIXES),
    'blocked_from_app_resolved_set': blocked,
    'resolved_application_count': len(reqs),
}, indent=2, sort_keys=True) + '\n')
if blocked:
    names = ', '.join(f"{item['normalized']}=={item['version']}" for item in blocked)
    raise SystemExit(f'protected NVIDIA base packages resolved as app deps: {names}; audit={AUDIT}')

TARGET.write_text('\n'.join(reqs) + '\n')
print(f'resolved {len(reqs)} application packages into {TARGET}; protected audit={AUDIT}')
