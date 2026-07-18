#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import torchao_model_cache


def sparse_write(path: Path, size: int, fill_tail: bytes = b"x") -> None:
    with path.open("wb") as f:
        if size > len(fill_tail):
            f.seek(size - len(fill_tail))
        f.write(fill_tail)


def measure(label: str, fn, repeat: int) -> dict:
    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return {
        "label": label,
        "result": bool(result),
        "repeat": repeat,
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "max_s": max(samples),
    }


def measure_fresh_metadata(label: str, make_metadata, fn, repeat: int) -> dict:
    samples = []
    result = None
    for _ in range(repeat):
        metadata = make_metadata()
        start = time.perf_counter()
        result = fn(metadata)
        samples.append(time.perf_counter() - start)
    return {
        "label": label,
        "result": bool(result),
        "repeat": repeat,
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "max_s": max(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark TorchAO cache metadata fast path vs strong-hash validation")
    parser.add_argument("--size-mb", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    size = args.size_mb * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="torchao-cache-bench-") as tmp:
        source = Path(tmp) / "large.safetensors"
        sparse_write(source, size)
        metadata = torchao_model_cache.stat_source(str(source))

        fast = measure("unchanged_identity_fast_path", lambda: torchao_model_cache.file_metadata_matches(str(source), metadata), args.repeat)

        changed_mtime = metadata["mtime_ns"] + 1_000_000_000
        os.utime(source, ns=(changed_mtime, changed_mtime))
        rehash = measure_fresh_metadata(
            "changed_mtime_strong_hash_path",
            lambda: dict(metadata),
            lambda stale_metadata: torchao_model_cache.file_metadata_matches(str(source), stale_metadata),
            args.repeat,
        )

        source.write_bytes(b"truncated")
        reject = measure("truncated_reject_path", lambda: torchao_model_cache.file_metadata_matches(str(source), metadata), args.repeat)

        print(json.dumps({"size_mb": args.size_mb, "results": [fast, rehash, reject]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
