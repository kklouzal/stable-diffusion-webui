from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "gb10" / "patch-controlnet-zoedepth.py"


def test_patcher_allows_only_derived_timm_position_indices(tmp_path):
    init_path = tmp_path / "sd-webui-controlnet" / "annotator" / "zoe" / "__init__.py"
    attractor_path = init_path.parent / "zoedepth" / "models" / "layers" / "attractor.py"
    dpt_path = (
        init_path.parent
        / "zoedepth"
        / "models"
        / "base_models"
        / "midas_repo"
        / "midas"
        / "dpt_depth.py"
    )
    attractor_path.parent.mkdir(parents=True)
    dpt_path.parent.mkdir(parents=True)
    init_path.write_text("            model.load_state_dict(torch.load(modelpath, map_location=model.device)['model'])\n")
    attractor_path.write_text("def exp_attractor():\n    pass\ndef inv_attractor():\n    pass\n")
    dpt_path.write_text("from timm.layers import get_act_layer\n")

    subprocess.run([sys.executable, str(PATCHER), str(init_path)], check=True)
    patched = init_path.read_text()
    assert "strict=False" in patched
    assert 'key.endswith(".attn.relative_position_index")' in patched
    assert "unsupported_missing" in patched
    assert "unsupported_unexpected" in patched
    subprocess.run([sys.executable, str(PATCHER), str(init_path)], check=True)


def test_deployment_applies_zoe_patch_after_extension_sync():
    run = (ROOT / "gb10" / "run.sh").read_text()
    sync = run.index('"${owned_extension_source}/" "${owned_extension_target}/"')
    patch = run.index("patch-controlnet-zoedepth.py")
    docker = run.index('DOCKER_ARGS=(')
    assert sync < patch < docker


def test_attention_projection_boundary_normalizes_strided_context():
    source = (ROOT / "modules" / "hypernetworks" / "hypernetwork.py").read_text()
    assert "return context_k.contiguous(), context_v.contiguous()" in source
