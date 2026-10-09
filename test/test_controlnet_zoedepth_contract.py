from pathlib import Path

import numpy as np

ZOE = Path(__file__).resolve().parents[1] / "extensions" / "sd-webui-controlnet" / "annotator" / "zoe"


def _zoe_detector_class(*methods, **namespace):
    """ZoeDetector with only the named methods of the real annotator/zoe/__init__.py (its imports need the zoedepth
    package and the webui runtime)."""
    import ast

    path = ZOE / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    detector = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ZoeDetector")
    body = [node for node in detector.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert len(body) == len(methods)
    module = ast.Module(body=[ast.ClassDef(name="ZoeDetector", bases=[], keywords=[], body=body, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["ZoeDetector"]


def test_zoedepth_loader_allows_only_derived_timm_position_indices(tmp_path):
    import os
    from types import SimpleNamespace

    import pytest

    loads = []

    class Model:
        device = "cpu"

        def __init__(self, missing, unexpected):
            self.incompatible = SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected)

        def load_state_dict(self, state, strict=True):
            loads.append((state, strict))
            return self.incompatible

        def eval(self):
            return self

        def to(self, device):
            return self

    def load(missing=(), unexpected=()):
        model = Model(list(missing), list(unexpected))
        torch_stub = SimpleNamespace(load=lambda path, map_location: {"model": ("weights", path, map_location)})
        detector_class = _zoe_detector_class("load_model", os=os, torch=torch_stub, get_config=lambda *args: "config",
                                             ZoeDepth=SimpleNamespace(build_from_config=lambda config: model))
        detector = detector_class()
        detector.model_dir, detector.device, detector.model = str(tmp_path), "cpu", None
        detector.load_model()
        return detector, model

    (tmp_path / "ZoeD_M12_N.pt").write_bytes(b"checkpoint")
    # Newer timm derives relative_position_index buffers instead of loading them: those checkpoint keys are expected.
    detector, model = load(unexpected=["core.core.pretrained.model.blocks.0.attn.relative_position_index"])
    assert detector.model is model
    assert loads == [(("weights", str(tmp_path / "ZoeD_M12_N.pt"), "cpu"), False)]
    # Any other mismatch still fails the load.
    for mismatch in ({"missing": ["core.core.pretrained.model.blocks.0.attn.qkv.weight"]},
                     {"unexpected": ["core.core.pretrained.model.blocks.0.attn.relative_position_bias_table"]}):
        with pytest.raises(RuntimeError, match="Unsupported ZoeDepth checkpoint mismatch"):
            load(**mismatch)


def test_zoedepth_sources_avoid_deprecated_torch_and_timm_apis():
    import ast
    import importlib.util
    import inspect
    import warnings

    import torch

    # The attractors are plain functions (torch.jit.script is deprecated), loaded without a DeprecationWarning.
    path = ZOE / "zoedepth" / "models" / "layers" / "attractor.py"
    spec = importlib.util.spec_from_file_location("zoe_attractor_under_test", path)
    attractor = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        spec.loader.exec_module(attractor)
    dx = torch.linspace(-0.2, 0.2, 9)
    assert inspect.isfunction(attractor.exp_attractor) and inspect.isfunction(attractor.inv_attractor)
    assert torch.equal(attractor.exp_attractor(dx), torch.exp(-300 * (torch.abs(dx) ** 2)) * dx)
    assert torch.equal(attractor.inv_attractor(dx), dx.div(1 + 300 * dx.pow(2)))

    # No zoe source imports the deprecated timm.models.layers alias or scripts a function.
    imports_get_act_layer = False
    for source in ZOE.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("timm.models.layers"), source
                imports_get_act_layer |= node.module == "timm.layers" and "get_act_layer" in {alias.name for alias in node.names}
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith("timm.models.layers") for alias in node.names), source
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert "torch.jit.script" not in {ast.unparse(decorator) for decorator in node.decorator_list}, source
    assert imports_get_act_layer


def _zoe_depth_image_two_percentile_calls(depth):
    """The depth image with the two percentile calls used before the single-pass change (rounded to uint8 as the
    detector does)."""
    depth = depth.copy()
    vmin = np.percentile(depth, 2)
    vmax = np.percentile(depth, 85)
    depth -= vmin
    depth /= vmax - vmin
    depth = 1.0 - depth
    return np.rint((depth * 255.0).clip(0, 255)).astype(np.uint8)


def test_zoedepth_single_percentile_pass_is_bitwise_identical_to_two_calls():
    from types import SimpleNamespace

    import torch
    from einops import rearrange

    detector = _zoe_detector_class("__call__", torch=torch, np=np, rearrange=rearrange)()
    detector.device = "cpu"
    rng = np.random.default_rng(0)
    for i in range(200):
        shape = (int(rng.integers(1, 300)), int(rng.integers(1, 300)))
        depth = (rng.random(shape) * rng.uniform(0.1, 20.0)).astype(np.float32)
        if i % 3 == 0:
            depth = np.round(depth, 1)  # ties
        prediction = torch.from_numpy(depth.copy())[None, None]
        detector.model = SimpleNamespace(to=lambda device: None, infer=lambda image, prediction=prediction: prediction)
        one = detector(np.zeros((*shape, 3), dtype=np.uint8))
        two = _zoe_depth_image_two_percentile_calls(depth)
        assert one.dtype == two.dtype
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
