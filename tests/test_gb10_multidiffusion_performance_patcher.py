"""gb10/patch-multidiffusion-performance.py: patcher contract plus CPU differential tests against the unpatched extension.

The extension is not part of the fork. These tests copy the deployed host checkout and skip when it is not mounted.
"""
from __future__ import annotations

import contextlib
import importlib.util
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
PATCHER = ROOT / "gb10" / "patch-multidiffusion-performance.py"
INSTALLED_MD = Path("/opt/gb10/stable-diffusion/Extensions/multidiffusion-upscaler-for-automatic1111")
SGM_ROOT = ROOT / "repositories" / "generative-models"


def load_patcher():
    spec = importlib.util.spec_from_file_location("gb10_patch_multidiffusion_performance", PATCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATCHER_MODULE = load_patcher()
TARGETS = list(PATCHER_MODULE.BLOCKS)


def run_patcher(root: Path, *extra: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PATCHER), *extra, str(root)], check=check, capture_output=True, text=True)


def copy_multidiffusion(target: Path) -> Path:
    if not INSTALLED_MD.exists():
        pytest.skip(f"installed MultiDiffusion fixture missing: {INSTALLED_MD}")
    shutil.copytree(INSTALLED_MD, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    return target


def snapshot(root: Path) -> dict[str, bytes]:
    return {relative: (root / relative).read_bytes() for relative in TARGETS}


# ---------------------------------------------------------------- patcher contract


def test_blocks_are_unambiguous():
    for relative, blocks in PATCHER_MODULE.BLOCKS.items():
        for name, original, patched in blocks:
            assert original not in patched and patched not in original, (relative, name)
            assert original.endswith("\n") and patched.endswith("\n"), (relative, name)


def test_patcher_is_idempotent_and_check_mode_verifies(tmp_path: Path):
    root = copy_multidiffusion(tmp_path / "md")
    (root / "scripts" / "tilevae.py").chmod(0o640)

    missing = run_patcher(root, "--check", check=False)
    assert missing.returncode != 0 and "performance patch missing" in missing.stderr
    first = run_patcher(root)
    assert first.stdout.count("Patched MultiDiffusion performance changes") == len(TARGETS)
    patched = snapshot(root)
    second = run_patcher(root)
    assert "already patched" in second.stdout
    assert snapshot(root) == patched
    assert "verified" in run_patcher(root, "--check").stdout

    assert (root / "scripts" / "tilevae.py").stat().st_mode & 0o777 == 0o640
    assert not list(root.rglob("*.gb10-tmp"))
    for relative in TARGETS:
        source = patched[relative].decode("utf-8")
        compile(source, relative, "exec")
        for _name, original, new in PATCHER_MODULE.BLOCKS[relative]:
            assert source.count(new) == 1 and original not in source


def test_patcher_fails_closed_without_writing_anything(tmp_path: Path):
    root = copy_multidiffusion(tmp_path / "md")
    tilevae = root / "scripts" / "tilevae.py"

    # Upstream drift in the last target must leave every earlier target unwritten too.
    pristine = snapshot(root)
    last = TARGETS[-1]
    name, original, _patched = PATCHER_MODULE.BLOCKS[last][-1]
    drifted = pristine[last].decode("utf-8").replace(original, original.replace("\n", " \n", 1)).encode("utf-8")
    (root / last).write_bytes(drifted)
    result = run_patcher(root, check=False)
    assert result.returncode != 0 and f"unsupported MultiDiffusion source for {name}" in result.stderr
    assert snapshot(root) == {**pristine, last: drifted}
    (root / last).write_bytes(pristine[last])

    tilevae.write_bytes(pristine["scripts/tilevae.py"].replace(b"\n", b"\r\n"))
    result = run_patcher(root, check=False)
    assert result.returncode != 0 and "CRLF" in result.stderr

    tilevae.write_bytes(pristine["scripts/tilevae.py"])
    run_patcher(root)
    name, original, patched = PATCHER_MODULE.BLOCKS["scripts/tilevae.py"][0]
    partial = tilevae.read_text(encoding="utf-8").replace(patched, original)
    tilevae.write_text(partial, encoding="utf-8")
    for extra in ((), ("--check",)):
        result = run_patcher(root, *extra, check=False)
        assert result.returncode != 0 and "partially patched" in result.stderr
    assert tilevae.read_text(encoding="utf-8") == partial


def test_run_sh_applies_and_checks_after_terminal_tiles_inside_the_multidiffusion_block():
    run_sh = (ROOT / "gb10" / "run.sh").read_text(encoding="utf-8")
    block_start = run_sh.index('if [[ -d "${MULTIDIFFUSION_ROOT}" ]]; then')
    block_end = run_sh.index("fi\n", block_start)
    block = run_sh[block_start:block_end]
    terminal = block.index("patch-multidiffusion-terminal-tiles.py")
    apply = block.index('gb10/patch-multidiffusion-performance.py" "${MULTIDIFFUSION_ROOT}"')
    verify = block.index('gb10/patch-multidiffusion-performance.py" --check "${MULTIDIFFUSION_ROOT}"')
    assert terminal < apply < verify


# ---------------------------------------------------------------- differential tests (CPU)


class NansException(Exception):
    pass


@pytest.fixture()
def webui_stubs(monkeypatch):
    """Minimal webui/ldm/k-diffusion stand-ins for importing the extension; records torch_gc and approximation calls."""
    if not SGM_ROOT.exists():
        pytest.skip(f"generative-models repository missing: {SGM_ROOT}")
    monkeypatch.syspath_prepend(str(SGM_ROOT))
    import sgm.modules.diffusionmodules.model  # noqa: F401  (real VAE blocks, imported before the stubs)

    calls = {"torch_gc": 0, "approx": 0, "sdpa": []}
    cpu = torch.device("cpu")

    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)
        return module

    def torch_gc():
        calls["torch_gc"] += 1

    def test_for_nans(x, where):
        # Same contract as modules.devices.test_for_nans: only element [0, ..., 0] is inspected.
        if torch.isnan(x[(0,) * x.ndim]):
            raise NansException(f"A tensor with NaNs was produced in {where}.")

    def cheap_approximation(sample):
        calls["approx"] += 1
        return sample[:3] * 0.5 + 0.25

    def run_scaled_dot_product_attention(q, k, v, *, mask=None, is_causal=False, sdpa_backend_override=None):
        calls["sdpa"].append((q.dim(), sdpa_backend_override))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal)

    state = types.SimpleNamespace(interrupted=False, sampling_step=0, sampling_steps=4)
    sd_model = types.SimpleNamespace(cond_stage_key="txt", model=types.SimpleNamespace(conditioning_key="crossattn"))
    shared = types.SimpleNamespace(state=state, opts=types.SimpleNamespace(upcast_attn=False), sd_model=sd_model)

    stub("gradio")
    stub("gradio.components", Component=object)
    stub("modules", __path__=[])
    stub("modules.scripts", Script=object, AlwaysVisible=object())
    stub("modules.devices", device=cpu, cpu=cpu, get_optimal_device=lambda: cpu, get_optimal_device_name=lambda: "cpu",
         torch_gc=torch_gc, autocast=contextlib.nullcontext, test_for_nans=test_for_nans, NansException=NansException)
    stub("modules.shared", state=state, opts=shared.opts, sd_model=sd_model, cmd_opts=types.SimpleNamespace())
    stub("modules.ui", gr_show=lambda *_args, **_kwargs: None)
    stub("modules.processing", opt_f=8, StableDiffusionProcessing=object, StableDiffusionProcessingImg2Img=object, Processed=object)
    stub("modules.sd_vae_approx", cheap_approximation=cheap_approximation)
    stub("modules.sd_hijack", model_hijack=types.SimpleNamespace(optimization_method="none"))
    stub("modules.sd_hijack_optimizations", get_available_vram=lambda: 2**40, get_xformers_flash_attention_op=lambda *_args: None,
         sub_quad_attention=None, run_scaled_dot_product_attention=run_scaled_dot_product_attention)
    stub("modules.prompt_parser", MulticondLearnedConditioning=object, ScheduledPromptConditioning=object)
    stub("modules.extra_networks", ExtraNetworkParams=object)
    stub("modules.sd_samplers_common")
    stub("modules.sd_samplers_kdiffusion", KDiffusionSampler=type("KDiffusionSampler", (), {}), CFGDenoiser=object, CFGDenoiserKDiffusion=object)
    stub("modules.sd_samplers_timesteps", CompVisSampler=type("CompVisSampler", (), {}), CFGDenoiserTimesteps=object,
         CompVisTimestepsDenoiser=object, CompVisTimestepsVDenoiser=object)
    stub("modules.shared_state", State=object)
    for package in ("ldm", "ldm.modules", "ldm.modules.diffusionmodules", "ldm.models", "ldm.models.diffusion", "k_diffusion"):
        stub(package, __path__=[])
    stub("ldm.modules.diffusionmodules.model", AttnBlock=object, MemoryEfficientAttnBlock=object)
    stub("ldm.models.diffusion.ddpm", LatentDiffusion=type("LatentDiffusion", (), {"apply_model": lambda *_args: None}))
    stub("k_diffusion.external", CompVisDenoiser=object, CompVisVDenoiser=object)
    return types.SimpleNamespace(state=state, calls=calls, shared=sys.modules["modules.shared"], hijack=sys.modules["modules.sd_hijack"])


def evict_extension_modules():
    for name in [name for name in sys.modules if name.split(".")[0] in ("tile_utils", "tile_methods")]:
        del sys.modules[name]


@pytest.fixture()
def import_extension(webui_stubs):
    """Import modules from one extension checkout; the extension's own packages are evicted before and after."""

    def load(root: Path, *names: str):
        evict_extension_modules()
        sys.path.insert(0, str(root))
        try:
            modules = []
            for name in names:
                if name.startswith("scripts/"):
                    spec = importlib.util.spec_from_file_location(f"md_{root.name}_{Path(name).stem}", root / name)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                else:
                    module = importlib.import_module(name)
                modules.append(module)
            return modules
        finally:
            sys.path.remove(str(root))

    yield load
    evict_extension_modules()


@pytest.fixture()
def md_pair(tmp_path: Path):
    original = copy_multidiffusion(tmp_path / "original")
    patched = copy_multidiffusion(tmp_path / "patched")
    run_patcher(patched)
    return original, patched


@pytest.fixture()
def tilevae_pair(md_pair, import_extension):
    original, patched = md_pair
    (old,) = import_extension(original, "scripts/tilevae.py")
    (new,) = import_extension(patched, "scripts/tilevae.py")
    return old, new


def tiny_vae(dtype=torch.float32):
    """A randomly initialized SDXL-architecture VAE (sgm Encoder/Decoder: 8x scale, mid attention, 32-group norms)."""
    from sgm.modules.diffusionmodules.model import Decoder, Encoder

    torch.manual_seed(0)
    config = dict(ch=32, out_ch=3, ch_mult=(1, 1, 1, 1), num_res_blocks=1, attn_resolutions=[], in_channels=3, resolution=64, z_channels=4)
    encoder = Encoder(**config).eval()
    decoder = Decoder(**config).eval()
    with torch.no_grad():
        # Non-trivial norm affine parameters, so a wrong statistic or a skipped affine shows up.
        for module in [*encoder.modules(), *decoder.modules()]:
            if isinstance(module, torch.nn.GroupNorm):
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.2, 0.2)
    for net in (encoder, decoder):
        net.original_forward = net.forward
    return encoder.to(dtype), decoder.to(dtype)


def run_hook(module, net, x, *, is_decoder, tile_size, fast, color_fix=False):
    hook = module.VAEHook(net, tile_size, is_decoder=is_decoder, fast_decoder=fast, fast_encoder=fast, color_fix=color_fix)
    return hook(x)


class CountCpuCopies:
    def __init__(self, monkeypatch):
        self.count = 0
        original = torch.Tensor.cpu

        def counting_cpu(tensor, *args, **kwargs):
            self.count += 1
            return original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)


CASES = [
    # (is_decoder, input shape, tile size, fast, color_fix)
    pytest.param(True, (1, 4, 40, 48), 12, True, False, id="decoder-fast"),
    pytest.param(True, (2, 4, 40, 47), 12, False, False, id="decoder-groupnorm-sync-batch2"),
    pytest.param(False, (1, 3, 192, 160), 64, True, False, id="encoder-fast"),
    pytest.param(False, (1, 3, 192, 168), 64, True, True, id="encoder-fast-colorfix"),
    pytest.param(False, (1, 3, 200, 160), 64, False, False, id="encoder-groupnorm-sync"),
]


@pytest.mark.parametrize("is_decoder,shape,tile_size,fast,color_fix", CASES)
def test_tiled_vae_is_bit_identical(tilevae_pair, webui_stubs, monkeypatch, is_decoder, shape, tile_size, fast, color_fix):
    old, new = tilevae_pair
    encoder, decoder = tiny_vae()
    net = decoder if is_decoder else encoder
    torch.manual_seed(1)
    x = torch.randn(shape)

    expected = run_hook(old, net, x.clone(), is_decoder=is_decoder, tile_size=tile_size, fast=fast, color_fix=color_fix)
    gc_calls = webui_stubs.calls["torch_gc"]
    assert gc_calls == 2
    copies = CountCpuCopies(monkeypatch)
    actual = run_hook(new, net, x.clone(), is_decoder=is_decoder, tile_size=tile_size, fast=fast, color_fix=color_fix)

    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual, expected)
    assert expected.shape[-1] == (shape[-1] * 8 if is_decoder else shape[-1] // 8)
    assert webui_stubs.calls["torch_gc"] == gc_calls
    assert copies.count == 0
    # The old code built the interrupt fallback eagerly, once per batch item; the new code only on interrupt.
    assert webui_stubs.calls["approx"] == shape[0]


def test_tiled_vae_bf16_result_dtype_is_bit_identical(tilevae_pair, webui_stubs):
    old, new = tilevae_pair
    _encoder, decoder = tiny_vae(torch.bfloat16)
    torch.manual_seed(2)
    z = torch.randn((1, 4, 40, 48), dtype=torch.bfloat16)
    expected = run_hook(old, decoder, z.clone(), is_decoder=True, tile_size=12, fast=True)
    actual = run_hook(new, decoder, z.clone(), is_decoder=True, tile_size=12, fast=True)
    assert expected.dtype == actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def test_tiled_vae_interrupt_returns_the_same_lazily_built_approximation(tilevae_pair, webui_stubs):
    old, new = tilevae_pair
    _encoder, decoder = tiny_vae()
    z = torch.randn((2, 4, 40, 48))
    webui_stubs.state.interrupted = True
    expected = run_hook(old, decoder, z.clone(), is_decoder=True, tile_size=12, fast=True)
    assert webui_stubs.calls["approx"] == 2
    actual = run_hook(new, decoder, z.clone(), is_decoder=True, tile_size=12, fast=True)
    assert webui_stubs.calls["approx"] == 4
    assert actual.shape == (2, 3, 320, 384)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("module_path,position", [
    ("conv_in", (0, 0, 0, 0)),
    ("conv_in", (0, 7, 5, 3)),
    ("mid.block_1.conv1", (0, 3, 2, 2)),
    ("mid.attn_1.proj_out", (0, 0, 1, 1)),
    ("up.2.upsample.conv", (0, 31, 0, 0)),
    ("up.0.block.0.conv2", (0, 0, 0, 0)),
    # Spreads to [0,0,0,0] through the next group norm and conv.
    ("up.0.block.0.conv2", (0, 9, 4, 4)),
    # The last op before the final norm: [0,0,0,0] stays finite, so neither version disables fast mode.
    ("up.0.block.1.conv2", (0, 9, 4, 4)),
    (None, None),
])
def test_tiled_vae_fast_mode_nan_decision_is_unchanged(tilevae_pair, webui_stubs, monkeypatch, module_path, position):
    old, new = tilevae_pair
    _encoder, decoder = tiny_vae()
    if module_path is not None:
        layer = decoder.get_submodule(module_path)
        forward = layer.forward

        def inject_nan(x):
            out = forward(x)
            out[position] = float("nan")
            return out

        monkeypatch.setattr(layer, "forward", inject_nan)
    torch.manual_seed(3)
    z = torch.randn((1, 4, 16, 16))
    decisions = []
    for module in (old, new):
        hook = module.VAEHook(decoder, 16, is_decoder=True, fast_decoder=True, fast_encoder=True, color_fix=False)
        queue = module.clone_task_queue(module.build_task_queue(decoder, True))
        decisions.append(hook.estimate_group_norm(z.clone(), queue, color_fix=False))
    assert decisions[0] == decisions[1]
    if position == (0, 0, 0, 0) or module_path == "up.0.block.0.conv2":
        assert decisions == [False, False]
    if module_path in (None, "up.0.block.1.conv2"):
        assert decisions == [True, True]


def test_tiled_vae_main_loop_nan_check_still_raises(tilevae_pair, webui_stubs):
    old, new = tilevae_pair
    _encoder, decoder = tiny_vae()
    z = torch.randn((1, 4, 40, 48))
    z[0, 0, 0, 0] = float("nan")
    for module in (old, new):
        with pytest.raises(NansException):
            run_hook(module, decoder, z.clone(), is_decoder=True, tile_size=12, fast=True)
