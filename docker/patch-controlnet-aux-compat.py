#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import py_compile
import subprocess
import sys


def main() -> int:
    spec = importlib.util.find_spec('controlnet_aux')
    if spec is None or spec.submodule_search_locations is None:
        raise SystemExit('controlnet_aux is not installed; cannot apply compatibility patch')
    root = pathlib.Path(next(iter(spec.submodule_search_locations))).resolve()
    changed: list[str] = []
    replacements = {
        '@torch.jit.script\n': '',
        '@register_tiny_vit_model\n': '',
        'from timm.models.layers import ': 'from timm.layers import ',
        'from timm.models.registry import ': 'from timm.models import ',
        'from scipy.ndimage.filters import ': 'from scipy.ndimage import ',
    }
    for path in root.rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        patched = text
        for old, new in replacements.items():
            patched = patched.replace(old, new)
        if patched != text:
            path.write_text(patched, encoding='utf-8')
            # A stale timestamp pyc would be recompiled in memory on every start (the runtime user cannot rewrite it).
            py_compile.compile(str(path), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
            changed.append(str(path.relative_to(root)))

    init_path = root / '__init__.py'
    init_text = init_path.read_text(encoding='utf-8')
    old = 'from .mediapipe_face import MediapipeFaceDetector\n'
    new = '''try:\n    import mediapipe as _gb10_mediapipe\n    if not hasattr(_gb10_mediapipe, "solutions"):\n        raise ImportError("mediapipe solutions API unavailable")\n    from .mediapipe_face import MediapipeFaceDetector\nexcept Exception as _gb10_mediapipe_exc:\n    class MediapipeFaceDetector:\n        unavailable_reason = str(_gb10_mediapipe_exc)\n\n        @classmethod\n        def from_pretrained(cls, *args, **kwargs):\n            raise RuntimeError("MediapipeFaceDetector unavailable: " + cls.unavailable_reason)\n\n'''
    if old in init_text and 'unavailable_reason = str(_gb10_mediapipe_exc)' not in init_text:
        init_path.write_text(init_text.replace(old, new), encoding='utf-8')
        py_compile.compile(str(init_path), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        changed.append('__init__.py:mediapipe_face_guard')

    print(f'controlnet_aux compatibility patch: files={len(changed)}')
    subprocess.run([sys.executable, '-c', "import controlnet_aux; print('controlnet_aux import clean')"], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
