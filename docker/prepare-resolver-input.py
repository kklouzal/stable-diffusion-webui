#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--wheel-dir', required=True)
    args = ap.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    wheel_dir = Path(args.wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)

    lines = source.read_text().splitlines()
    gradio_line = next((line.strip() for line in lines if line.strip().startswith('gradio==')), None)
    if not gradio_line:
        target.write_text(source.read_text())
        print(f'no gradio pin found in {source}; wrote passthrough {target}')
        return 0

    m = re.fullmatch(r'gradio==(.+)', gradio_line)
    if not m:
        raise SystemExit(f'unexpected gradio requirement format: {gradio_line}')
    version = m.group(1)

    subprocess.run([
        sys.executable, '-m', 'pip', 'download', '--no-deps', '--dest', str(wheel_dir), f'gradio=={version}'
    ], check=True)

    wheels = sorted(wheel_dir.glob(f'gradio-{version}-*.whl'))
    if not wheels:
        raise SystemExit(f'failed to download gradio wheel for version {version}')
    wheel = wheels[0]
    patched = wheel_dir / wheel.name.replace('.whl', '.numpy2compat.whl')

    replaced = False
    with zipfile.ZipFile(wheel) as zin, zipfile.ZipFile(patched, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.endswith('METADATA'):
                text = data.decode()
                old = 'Requires-Dist: numpy~=1.0\n'
                new = 'Requires-Dist: numpy>=1.0\n'
                if old in text:
                    text = text.replace(old, new)
                    replaced = True
                data = text.encode()
            zout.writestr(info, data)

    if not replaced:
        raise SystemExit('failed to patch gradio wheel metadata; numpy~=1.0 line not found')

    out_lines = []
    replaced_req = False
    for raw in lines:
        if raw.strip() == gradio_line and not replaced_req:
            out_lines.append(str(patched))
            replaced_req = True
        else:
            out_lines.append(raw)
    target.write_text('\n'.join(out_lines) + '\n')
    print(f'prepared resolver input {target} using patched wheel {patched.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
