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


def _beit_get_rel_pos_bias():
    """The real _get_rel_pos_bias from the vendored MiDaS BEiT backbone (its module needs the zoedepth package)."""
    import ast

    import torch
    import torch.nn.functional as F
    from timm.models.beit import gen_relative_position_index

    path = ZOE / "zoedepth" / "models" / "base_models" / "midas_repo" / "midas" / "backbones" / "beit.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_get_rel_pos_bias")
    namespace = {"torch": torch, "F": F, "gen_relative_position_index": gen_relative_position_index}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_get_rel_pos_bias"], gen_relative_position_index


def test_beit_relative_position_index_is_cached_on_the_bias_table_device():
    """depth_zoe's BEiT indexed its CUDA bias table with a CPU index (~8 MB per block per call at 512x512):
    the index is now kept on the table's device. Same gather, same bias."""
    import types

    import torch
    import torch.nn.functional as F

    get_rel_pos_bias, gen_relative_position_index = _beit_get_rel_pos_bias()
    window, new_window, heads = (4, 4), (6, 5), 3
    num_relative_distance = (2 * window[0] - 1) * (2 * window[1] - 1) + 3
    table = torch.randn(num_relative_distance, heads, generator=torch.Generator().manual_seed(0))

    def attn(table):
        return types.SimpleNamespace(window_size=window, num_relative_distance=num_relative_distance,
                                     relative_position_bias_table=table, relative_position_indices={})

    # Former code path, written out: interpolated table gathered with a freshly generated CPU index.
    old_h, old_w, new_h, new_w = 2 * window[0] - 1, 2 * window[1] - 1, 2 * new_window[0] - 1, 2 * new_window[1] - 1
    sub = table[:num_relative_distance - 3].reshape(1, old_w, old_h, -1).permute(0, 3, 1, 2)
    sub = F.interpolate(sub, size=(new_h, new_w), mode="bilinear").permute(0, 2, 3, 1).reshape(new_h * new_w, -1)
    new_table = torch.cat([sub, table[num_relative_distance - 3:]])
    n = new_window[0] * new_window[1] + 1
    expected = new_table[gen_relative_position_index(new_window).view(-1)].view(n, n, -1).permute(2, 0, 1).contiguous().unsqueeze(0)

    cpu = attn(table)
    first = get_rel_pos_bias(cpu, new_window)
    index = cpu.relative_position_indices["5,6"]
    second = get_rel_pos_bias(cpu, new_window)
    assert torch.equal(first, expected) and torch.equal(second, expected)
    assert cpu.relative_position_indices["5,6"] is index

    meta = attn(table.to("meta"))
    assert get_rel_pos_bias(meta, new_window).device.type == "meta"
    assert meta.relative_position_indices["5,6"].device.type == "meta"
