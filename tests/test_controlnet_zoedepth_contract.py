from pathlib import Path

import numpy as np

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


def _zoe_depth_image(depth, single_pass):
    depth = depth.copy()
    if single_pass:
        vmin, vmax = np.percentile(depth, np.array([2, 85], dtype=depth.dtype))
    else:
        vmin = np.percentile(depth, 2)
        vmax = np.percentile(depth, 85)
    depth -= vmin
    depth /= vmax - vmin
    depth = 1.0 - depth
    return (depth * 255.0).clip(0, 255).astype(np.uint8), vmin, vmax


def test_zoedepth_single_percentile_pass_is_bitwise_identical_to_two_calls():
    source = (ZOE / "__init__.py").read_text(encoding="utf-8")
    assert "vmin, vmax = np.percentile(depth, np.array([2, 85], dtype=depth.dtype))" in source
    rng = np.random.default_rng(0)
    for i in range(200):
        shape = (int(rng.integers(1, 300)), int(rng.integers(1, 300)))
        depth = (rng.random(shape) * rng.uniform(0.1, 20.0)).astype(np.float32)
        if i % 3 == 0:
            depth = np.round(depth, 1)  # ties
        one, vmin1, vmax1 = _zoe_depth_image(depth, single_pass=True)
        two, vmin2, vmax2 = _zoe_depth_image(depth, single_pass=False)
        assert type(vmin1) is type(vmin2) and type(vmax1) is type(vmax2)
        assert vmin1 == vmin2 and vmax1 == vmax2
        assert np.array_equal(one, two)
