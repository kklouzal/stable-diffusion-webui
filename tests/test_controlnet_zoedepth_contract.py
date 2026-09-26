from pathlib import Path

ZOE = Path(__file__).resolve().parents[1] / "extensions" / "sd-webui-controlnet" / "annotator" / "zoe"


def test_zoedepth_loader_allows_only_derived_timm_position_indices():
    source = (ZOE / "__init__.py").read_text(encoding="utf-8")
    assert "torch.load(modelpath, map_location=model.device)['model'], strict=False" in source
    assert 'key.endswith(".attn.relative_position_index")' in source
    assert "if unsupported_missing or unsupported_unexpected:" in source
    assert "model.load_state_dict(torch.load(modelpath, map_location=model.device)['model'])\n" not in source


def test_zoedepth_sources_avoid_deprecated_torch_and_timm_apis():
    attractor = (ZOE / "zoedepth" / "models" / "layers" / "attractor.py").read_text(encoding="utf-8")
    assert "@torch.jit.script" not in attractor
    assert "def exp_attractor" in attractor
    assert "def inv_attractor" in attractor

    dpt = (ZOE / "zoedepth" / "models" / "base_models" / "midas_repo" / "midas" / "dpt_depth.py").read_text(encoding="utf-8")
    assert "from timm.layers import get_act_layer" in dpt
    assert "from timm.models.layers import get_act_layer" not in dpt
