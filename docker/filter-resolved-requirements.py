#!/usr/bin/env python3
import os
import re
from pathlib import Path

SOURCE = Path(os.environ.get('SOURCE', '/opt/build/requirements-resolved.txt'))
TARGET = Path(os.environ.get('TARGET', '/opt/build/requirements-runtime.txt'))
BASE_PROTECTED_NAMES_FILE = Path(os.environ.get('BASE_PROTECTED_NAMES_FILE', '/opt/build/base-python-protected-names.txt'))

DROP_PREFIXES = (
    'torch==',
    'torchvision==',
    'torchaudio==',
    'triton==',
    'nvidia-',
    'cuda-',
)


def normalize_name(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name.strip().lower())


protected = set()
if BASE_PROTECTED_NAMES_FILE.exists():
    protected = {
        normalize_name(line)
        for line in BASE_PROTECTED_NAMES_FILE.read_text().splitlines()
        if line.strip()
    }

lines = []
for raw in SOURCE.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    lower = line.lower()
    if lower.startswith(DROP_PREFIXES):
        continue
    name = normalize_name(re.split(r'[<>=!~ \[;]', line, maxsplit=1)[0])
    if name in protected:
        continue
    lines.append(line)

TARGET.write_text('\n'.join(lines) + '\n')
print(f'wrote {TARGET} with {len(lines)} filtered requirements; protected names={len(protected)} source={SOURCE}')
