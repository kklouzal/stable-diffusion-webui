#!/usr/bin/env python3
"""Patch and verify Ultimate Upscale state lifecycle."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

TARGET_RELATIVE = Path("scripts") / "ultimate-upscale.py"
MARKER = "OPENCLAW_ULTIMATE_UPSCALE_STATE_FINALLY_V2"


def target_for(path: Path) -> Path:
    return path / TARGET_RELATIVE if path.is_dir() else path


def process_node(source: str, target: Path) -> ast.FunctionDef:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SystemExit(f"invalid Ultimate Upscale source: {target}: {exc}") from exc
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "process"]
    if len(matches) != 1:
        raise SystemExit(f"unsupported Ultimate Upscale process implementation: {target}")
    return matches[0]


def calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and isinstance(child.func.value, ast.Name)
        and child.func.value.id == "state"
        and child.func.attr == name
    ]


def verify(source: str, target: Path) -> None:
    node = process_node(source, target)
    if source.count(MARKER) != 1 or len(calls(node, "begin")) != 1 or len(calls(node, "end")) != 1:
        raise SystemExit(f"Ultimate Upscale lifecycle verification failed (partial markers): {target}")
    finalizers = [child for child in node.body if isinstance(child, ast.Try)]
    if len(finalizers) != 1 or len(finalizers[0].finalbody) != 1 or len(calls(finalizers[0].finalbody[0], "end")) != 1:
        raise SystemExit(f"Ultimate Upscale lifecycle verification failed (state.end not in finally): {target}")


def patch(source: str, target: Path) -> str:
    node = process_node(source, target)
    begins = calls(node, "begin")
    ends = calls(node, "end")
    if len(begins) != 1 or len(ends) != 1 or not node.body:
        raise SystemExit(f"unsupported or partial Ultimate Upscale state lifecycle: {target}")
    if not isinstance(node.body[0], ast.Expr) or node.body[0].value is not begins[0]:
        raise SystemExit(f"unsupported Ultimate Upscale state.begin placement: {target}")
    if not isinstance(node.body[-1], ast.Expr) or node.body[-1].value is not ends[0]:
        raise SystemExit(f"unsupported Ultimate Upscale state.end placement: {target}")

    lines = source.splitlines(keepends=True)
    indent = " " * (node.col_offset + 4)
    body_start = node.body[1].lineno - 1
    body_end = node.body[-1].lineno - 1
    body = lines[body_start:body_end]
    indented_body = [indent + "    " + line[len(indent) :] if line.strip() else line for line in body]
    replacement = [
        f"{indent}try:\n",
        f"{indent}    # {MARKER}: one begin owns one end.\n",
        *indented_body,
        f"{indent}finally:\n",
        f"{indent}    state.end()\n",
    ]
    return "".join(lines[:body_start] + replacement + lines[body_end + 1 :])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = target_for(args.path)
    source = target.read_text(encoding="utf-8")
    if not args.check and MARKER not in source:
        source = patch(source, target)
        target.write_text(source, encoding="utf-8")
        print(f"Patched Ultimate Upscale state lifecycle: {target}")
    verify(source, target)
    print(f"Ultimate Upscale lifecycle verified: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
