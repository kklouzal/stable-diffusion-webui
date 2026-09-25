#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

SOURCE = Path(os.environ.get('SOURCE', '/opt/build/requirements-resolved.txt'))
TARGET = Path(os.environ.get('TARGET', '/opt/build/requirements-runtime.txt'))
AUDIT = Path(os.environ.get('AUDIT', str(TARGET) + '.protected-audit.json'))
BASE_PROTECTED_NAMES_FILE = Path(os.environ.get('BASE_PROTECTED_NAMES_FILE', '/opt/build/base-python-protected-names.txt'))

ALWAYS_PROTECTED_NAMES = {
    'torch',
    'torchvision',
    'torchaudio',
    'triton',
}
ALWAYS_PROTECTED_PREFIXES = (
    'nvidia-',
    'cuda-',
)


def normalize_name(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name.strip().lower())


protected = set(ALWAYS_PROTECTED_NAMES)
if BASE_PROTECTED_NAMES_FILE.exists():
    protected.update(
        normalize_name(name)
        for name in BASE_PROTECTED_NAMES_FILE.read_text().splitlines()
        if name.strip() and not name.lstrip().startswith('#')
    )

lines = []
removed = []
for raw in SOURCE.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    name = normalize_name(re.split(r'[<>=!~ \[;]', line, maxsplit=1)[0])
    reason = None
    if name in protected:
        reason = 'protected installed/base package'
    elif name.startswith(ALWAYS_PROTECTED_PREFIXES):
        reason = 'protected CUDA/NVIDIA package prefix'
    if reason:
        removed.append({'name': name, 'line': line, 'reason': reason})
        continue
    lines.append(line)

TARGET.write_text('\n'.join(lines) + '\n')
AUDIT.write_text(json.dumps({
    'source': str(SOURCE),
    'target': str(TARGET),
    'base_protected_names_file': str(BASE_PROTECTED_NAMES_FILE),
    'base_protected_names_note': 'all NVIDIA base-image packages are protected from application replacement',
    'protected_names_count': len(protected),
    'always_protected_names': sorted(ALWAYS_PROTECTED_NAMES),
    'always_protected_prefixes': list(ALWAYS_PROTECTED_PREFIXES),
    'removed': removed,
    'kept_count': len(lines),
}, indent=2, sort_keys=True) + '\n')
print(f'wrote {TARGET} with {len(lines)} filtered requirements; protected names={len(protected)} source={SOURCE}; audit={AUDIT}')
if removed:
    print('filtered protected app requirements: ' + ', '.join(sorted({item['name'] for item in removed})))
