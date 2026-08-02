from pathlib import Path
import subprocess
import sys


def _write_legacy_zoedepth_fixture(root: Path) -> Path:
    zoe = root / "sd-webui-controlnet" / "annotator" / "zoe"
    (zoe / "zoedepth" / "models" / "layers").mkdir(parents=True)
    (zoe / "zoedepth" / "models" / "base_models" / "midas_repo" / "midas").mkdir(parents=True)
    init_path = zoe / "__init__.py"
    init_path.write_text(
        "import torch\n"
        "class ZoeDetector:\n"
        "    def load_model(self):\n"
        "        model.load_state_dict(torch.load(modelpath, map_location=model.device)['model'])\n",
        encoding="utf-8",
    )
    (zoe / "zoedepth" / "models" / "layers" / "attractor.py").write_text(
        "import torch\n\n"
        "@torch.jit.script\n"
        "def exp_attractor(dx, alpha: float = 300, gamma: int = 2):\n"
        "    return torch.exp(-alpha*(torch.abs(dx)**gamma)) * (dx)\n\n"
        "@torch.jit.script\n"
        "def inv_attractor(dx, alpha: float = 300, gamma: int = 2):\n"
        "    return dx.div(1+alpha*dx.pow(gamma))\n",
        encoding="utf-8",
    )
    (zoe / "zoedepth" / "models" / "base_models" / "midas_repo" / "midas" / "dpt_depth.py").write_text(
        "from timm.models.layers import get_act_layer\n",
        encoding="utf-8",
    )
    return init_path


def test_controlnet_zoedepth_runtime_patcher_removes_known_warning_sources(tmp_path):
    init_path = _write_legacy_zoedepth_fixture(tmp_path)
    attractor_path = init_path.parent / "zoedepth" / "models" / "layers" / "attractor.py"
    dpt_path = init_path.parent / "zoedepth" / "models" / "base_models" / "midas_repo" / "midas" / "dpt_depth.py"

    subprocess.run([sys.executable, "gb10/patch-controlnet-zoedepth.py", str(init_path)], check=True)

    assert "strict=False" in init_path.read_text(encoding="utf-8")
    assert "@torch.jit.script" not in attractor_path.read_text(encoding="utf-8")
    assert "def exp_attractor" in attractor_path.read_text(encoding="utf-8")
    assert "def inv_attractor" in attractor_path.read_text(encoding="utf-8")
    dpt_text = dpt_path.read_text(encoding="utf-8")
    assert "from timm.models.layers import get_act_layer" not in dpt_text
    assert "from timm.layers import get_act_layer" in dpt_text


def test_controlnet_zoedepth_runtime_patcher_is_idempotent(tmp_path):
    init_path = _write_legacy_zoedepth_fixture(tmp_path)
    subprocess.run([sys.executable, "gb10/patch-controlnet-zoedepth.py", str(init_path)], check=True)
    subprocess.run([sys.executable, "gb10/patch-controlnet-zoedepth.py", str(init_path)], check=True)


def test_run_sh_applies_controlnet_zoedepth_patch_to_mounted_production_extension():
    run_sh = Path("gb10/run.sh").read_text(encoding="utf-8")
    assert "patch-controlnet-zoedepth.py" in run_sh
    assert "annotator/zoe/__init__.py" in run_sh


def test_controlnet_zoedepth_patcher_documents_deprecated_sources():
    patcher = Path("gb10/patch-controlnet-zoedepth.py").read_text(encoding="utf-8")
    assert "@torch.jit.script" in patcher
    assert "from timm.models.layers import get_act_layer" in patcher
    assert "from timm.layers import get_act_layer" in patcher
    assert "strict=False" in patcher
