#!/usr/bin/env python3
"""Assert what the resolver dry run (pip --dry-run --report) selected for one package: present at or above a
version floor (optionally as a wheel), or absent. A package missing from the report fails unless --absent."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from packaging.utils import canonicalize_name
from packaging.version import Version


def main() -> int:
    ap = argparse.ArgumentParser(description='Validate a package selected by pip --dry-run --report.')
    ap.add_argument('--report', default='/opt/build/report.json')
    ap.add_argument('--package', required=True)
    ap.add_argument('--min-version')
    ap.add_argument('--absent', action='store_true', help='fail if the package is present in the pip report')
    ap.add_argument('--require-wheel', action='store_true')
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text())
    wanted = canonicalize_name(args.package)
    matches = [
        item
        for item in report.get('install', [])
        if (item.get('metadata') or {}).get('name') and canonicalize_name(item['metadata']['name']) == wanted
    ]

    if args.absent:
        if matches:
            versions = ', '.join((item.get('metadata') or {}).get('version', '<unknown>') for item in matches)
            raise SystemExit(f'{args.package}: unexpectedly present in pip report: {versions}')
        print(f'{args.package}: absent from pip report')
        return 0

    if len(matches) != 1:
        raise SystemExit(f'{args.package}: expected one pip report entry, found {len(matches)}')

    item = matches[0]
    version = (item.get('metadata') or {}).get('version')
    if not version:
        raise SystemExit(f'{args.package}: report entry has no version')
    if args.min_version and Version(version) < Version(args.min_version):
        raise SystemExit(f'{args.package}: resolved {version}, below required floor {args.min_version}')

    url = (item.get('download_info') or {}).get('url') or ''
    path = urlparse(url).path
    if args.require_wheel and not path.endswith('.whl'):
        raise SystemExit(f'{args.package}: resolved artifact is not a wheel: {url or "<missing url>"}')

    print(f'{args.package}: resolved {version}; artifact={Path(path).name or url or "<unknown>"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
