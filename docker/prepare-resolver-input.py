#!/usr/bin/env python3
import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--wheel-dir", required=True)
    ap.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional requirements files to append to the resolver input.",
    )
    args = ap.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    Path(args.wheel_dir).mkdir(parents=True, exist_ok=True)

    chunks = [source.read_text().rstrip()]
    included = []
    for include in args.include:
        include_path = Path(include)
        if not include_path.exists():
            raise FileNotFoundError(f"included requirements file not found: {include_path}")
        chunks.append(f"# requirements from {include_path}")
        chunks.append(include_path.read_text().rstrip())
        included.append(str(include_path))

    target.write_text("\n".join(chunk for chunk in chunks if chunk) + "\n")
    suffix = f" plus {included}" if included else ""
    print(f"wrote resolver input {target} from {source}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
