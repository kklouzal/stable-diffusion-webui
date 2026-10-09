#!/usr/bin/env python3
"""Patch and verify Ultimate Upscale state lifecycle.

USDUpscaler.process() calls state.begin() first and state.end() last; an exception in between left the shared job
state open. The patch wraps everything between them in try/finally so one begin always owns one end.
The rewrite is an AST-checked re-indent rather than exact text blocks, but it follows gb10/patchlib.py's contract:
UTF-8 with LF only, the rewrapped body must parse to exactly the original statements and the result must verify
before anything is written, the file is replaced atomically, and --check writes nothing.
"""
from __future__ import annotations

import ast
from pathlib import Path

from patchlib import parse_cli, read_lf, replace_atomically

LABEL = "Ultimate Upscale lifecycle"
TARGET_RELATIVE = Path("scripts") / "ultimate-upscale.py"
MARKER = "OPENCLAW_ULTIMATE_UPSCALE_STATE_FINALLY_V2"


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
    patched = "".join(lines[:body_start] + replacement + lines[body_end + 1 :])
    # Re-indenting text lines is only valid when every body line is indented code (no multi-line string or
    # continuation at a shallower indent); require the wrapped statements to parse to exactly the original ones.
    wrapped = [child for child in process_node(patched, target).body if isinstance(child, ast.Try)]
    if len(wrapped) != 1 or [ast.dump(stmt) for stmt in wrapped[0].body] != [ast.dump(stmt) for stmt in node.body[1:-1]]:
        raise SystemExit(f"unsupported Ultimate Upscale process body (re-indentation changed its statements): {target}")
    return patched


def main() -> int:
    args = parse_cli(__doc__.splitlines()[0])
    target = args.path / TARGET_RELATIVE if args.path.is_dir() else args.path
    source = read_lf(target, LABEL)
    if args.check or MARKER in source:
        verify(source, target)
        print(f"Ultimate Upscale lifecycle verified: {target}")
    else:
        source = patch(source, target)
        verify(source, target)
        replace_atomically(target, source)
        print(f"Patched Ultimate Upscale state lifecycle: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
