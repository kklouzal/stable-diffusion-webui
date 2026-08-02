#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit('usage: patch-controlnet-teed.py /path/to/sd-webui-controlnet')
    root = pathlib.Path(argv[1]).resolve() / 'annotator' / 'teed'
    if not root.exists():
        raise SystemExit(f'ControlNet TEED source is missing: {root}')

    changed: list[str] = []
    for name in ('Fsmish.py', 'Fmish.py'):
        path = root / name
        original = path.read_text(encoding='utf-8')
        patched = original.replace('@torch.jit.script\n', '')
        if patched != original:
            path.write_text(patched, encoding='utf-8')
            changed.append(str(path))

    if changed:
        print('Patched ControlNet TEED deprecated torch.jit.script decorators: ' + ', '.join(changed))
    else:
        print(f'ControlNet TEED torch.jit.script compatibility patch already present: {root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
