#!/usr/bin/env python3
import json
from pathlib import Path

REPORT = Path('/opt/build/report.json')
TARGET = Path('/opt/build/requirements-resolved.txt')

report = json.loads(REPORT.read_text())
reqs = []
for item in report.get('install', []):
    meta = item.get('metadata', {})
    name = meta.get('name')
    version = meta.get('version')
    if name and version:
        reqs.append(f'{name}=={version}')

TARGET.write_text('\n'.join(reqs) + '\n')
print(f'resolved {len(reqs)} packages into {TARGET}')
