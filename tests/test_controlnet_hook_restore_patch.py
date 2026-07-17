import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATCH = ROOT / "gb10" / "patch-controlnet-hook-restore.py"
SOURCE = ROOT / "extensions" / "sd-webui-controlnet" / "scripts" / "hook.py"


def test_controlnet_hook_restore_patch_is_idempotent(tmp_path):
    target = tmp_path / "hook.py"
    target.write_text(SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    first = subprocess.run([sys.executable, PATCH, target], check=True, capture_output=True, text=True)
    patched = target.read_text(encoding="utf-8")
    second = subprocess.run([sys.executable, PATCH, target], check=True, capture_output=True, text=True)
    assert "Patched ControlNet" in first.stdout or "already present" in first.stdout
    assert "already present" in second.stdout
    assert target.read_text(encoding="utf-8") == patched
    forward = patched.index("def forward_webui")
    assert patched.index("if outer.control_params is None:", forward) < patched.index("return forward(*args, **kwargs)", forward)
    assert "return outer.original_forward(*args, **kwargs)" in patched[forward:]
    restore = patched.index("def restore(self):")
    assert patched.index("if self.model is not None:", restore) < patched.index("self.control_params = None", restore)
