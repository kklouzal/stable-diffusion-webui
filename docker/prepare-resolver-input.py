#!/usr/bin/env python3
import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--wheel-dir", required=True)
    args = ap.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    Path(args.wheel_dir).mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text())
    print(f"wrote resolver input {target} from {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
