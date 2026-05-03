#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import re


def normalize(name: str) -> str:
    return name.replace('_', '-').lower()


def version_key(version: str) -> tuple:
    # Good enough for the numeric stable-version floors this guard enforces
    # (for example 5.7.0, 0.22.2, 1.13.0). Keep this script dependency-free
    # because it runs before the resolved application closure is installed.
    return tuple(int(part) for part in re.findall(r"\d+", version.split("+", 1)[0]))


def main() -> int:
    ap = argparse.ArgumentParser(description='Validate a package selected by pip --dry-run --report.')
    ap.add_argument('--report', default='/opt/build/report.json')
    ap.add_argument('--package', required=True)
    ap.add_argument('--min-version')
    ap.add_argument('--require-wheel', action='store_true')
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text())
    wanted = normalize(args.package)
    matches = []
    for item in report.get('install', []):
        meta = item.get('metadata') or {}
        name = meta.get('name')
        if name and normalize(name) == wanted:
            matches.append(item)

    if not matches:
        raise SystemExit(f'{args.package}: not present in pip report')
    if len(matches) != 1:
        raise SystemExit(f'{args.package}: expected one report entry, found {len(matches)}')

    item = matches[0]
    meta = item.get('metadata') or {}
    version = meta.get('version')
    if not version:
        raise SystemExit(f'{args.package}: report entry has no version')

    if args.min_version and version_key(version) < version_key(args.min_version):
        raise SystemExit(f'{args.package}: resolved {version}, below required floor {args.min_version}')

    url = ((item.get('download_info') or {}).get('url') or '')
    path = urlparse(url).path
    if args.require_wheel and not path.endswith('.whl'):
        raise SystemExit(f'{args.package}: resolved artifact is not a wheel: {url or "<missing url>"}')

    print(f'{args.package}: resolved {version}; artifact={Path(path).name or url or "<unknown>"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
