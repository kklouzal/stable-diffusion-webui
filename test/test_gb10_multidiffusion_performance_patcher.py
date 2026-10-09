"""gb10/patch-multidiffusion-performance.py: patcher contract plus CPU differential tests against the unpatched extension.

The extension is not part of the fork. These tests copy the installed checkout (the host deploy root, or the same
checkout where run.sh mounts it inside a webui container) and skip when neither is present. The installed checkout
may already be patched, so the fixture reverses every block to its upstream text and asserts the patcher reproduces
the installed bytes. The shared fail-closed engine itself is tested in test_gb10_patchlib.py.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import functools
import importlib.util
import math
import re
import resource
import shutil
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).parents[1]
GB10 = ROOT / "gb10"
PATCHER = GB10 / "patch-multidiffusion-performance.py"
EXTENSION = "multidiffusion-upscaler-for-automatic1111"
INSTALLED_MD = next((path for path in (Path("/opt/gb10/stable-diffusion/Extensions") / EXTENSION, ROOT / "extensions" / EXTENSION) if path.is_dir()), None)
UPSTREAM_COMMIT = "22798f6"  # origin/main the blocks' ORIGINAL texts are taken from
SGM_ROOT = ROOT / "repositories" / "generative-models"
UTILS = "tile_utils/utils.py"
TERMINAL_HELPER = "def _gb10_terminal_tile_origins"


def load_patcher():
    if str(GB10) not in sys.path:
        sys.path.insert(0, str(GB10))  # the patcher imports patchlib as a sibling module, as under run.sh
    spec = importlib.util.spec_from_file_location("gb10_patch_multidiffusion_performance", PATCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATCHER_MODULE = load_patcher()
TARGETS = list(PATCHER_MODULE.BLOCKS)


def run_patcher(root: Path, *extra: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PATCHER), *extra, str(root)], check=check, capture_output=True, text=True)


def snapshot(root: Path) -> dict[str, bytes]:
    return {relative: (root / relative).read_bytes() for relative in TARGETS}


def unpatched(source: str, blocks) -> str:
    for block in reversed(blocks):
        source = source.replace(block.patched, block.original)
    return source


def copy_multidiffusion(target: Path) -> Path:
    """Copy of the installed checkout with every target at its upstream text (a fresh install, as run.sh sees it)."""
    if INSTALLED_MD is None:
        pytest.skip(f"installed MultiDiffusion fixture missing: {EXTENSION}")
    shutil.copytree(INSTALLED_MD, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    installed = snapshot(target)
    for relative, blocks in PATCHER_MODULE.BLOCKS.items():
        (target / relative).write_bytes(unpatched(installed[relative].decode("utf-8"), blocks).encode("utf-8"))
    if snapshot(target) != installed:
        # The derived tree is only a valid upstream stand-in if the patcher turns it back into the installed bytes.
        probe = target.parent / f"{target.name}-roundtrip"
        for relative in TARGETS:
            (probe / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target / relative, probe / relative)
        run_patcher(probe)
        assert snapshot(probe) == installed
        shutil.rmtree(probe)
    return target


def apply_block(root: Path, relative: str, name: str) -> None:
    """Apply one named block by hand (no patcher): isolates the other blocks' changes in a differential."""
    (block,) = [block for block in PATCHER_MODULE.BLOCKS[relative] if block.name == name]
    path = root / relative
    path.write_text(path.read_text(encoding="utf-8").replace(block.original, block.patched), encoding="utf-8")


# ---------------------------------------------------------------- patcher contract


def test_blocks_are_unambiguous():
    for relative, blocks in PATCHER_MODULE.BLOCKS.items():
        texts = [text for block in blocks for text in (block.original, block.patched)]
        for block in blocks:
            assert block.original.endswith("\n") and block.patched.endswith("\n"), (relative, block.name)
        for i, a in enumerate(texts):
            for j, b in enumerate(texts):
                assert i == j or a not in b, (relative, i, j)


def test_derived_tree_is_the_pinned_upstream_commit(tmp_path: Path):
    """The reversed fixture is byte-identical to upstream origin/main, so the patcher accepts a fresh install."""
    root = copy_multidiffusion(tmp_path / "md")
    git = ["git", "-c", "safe.directory=*", "-C", str(INSTALLED_MD)]
    if subprocess.run([*git, "cat-file", "-e", f"{UPSTREAM_COMMIT}^{{commit}}"], capture_output=True).returncode != 0:
        pytest.skip(f"installed checkout has no git history with {UPSTREAM_COMMIT}")
    for relative in TARGETS:
        upstream = subprocess.run([*git, "show", f"{UPSTREAM_COMMIT}:{relative}"], check=True, capture_output=True).stdout
        assert (root / relative).read_bytes() == upstream, relative


def test_patcher_is_idempotent_and_check_mode_verifies(tmp_path: Path):
    root = copy_multidiffusion(tmp_path / "md")
    (root / "scripts" / "tilevae.py").chmod(0o640)

    missing = run_patcher(root, "--check", check=False)
    assert missing.returncode != 0 and "MultiDiffusion patch missing" in missing.stderr
    first = run_patcher(root)
    assert first.stdout.count("Patched MultiDiffusion changes") == len(TARGETS)
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
        for block in PATCHER_MODULE.BLOCKS[relative]:
            assert source.count(block.patched) == 1 and block.original not in source
    assert patched[UTILS].decode("utf-8").count(TERMINAL_HELPER) == 1


def test_patcher_fails_closed_without_writing_anything(tmp_path: Path):
    root = copy_multidiffusion(tmp_path / "md")
    tilevae = root / "scripts" / "tilevae.py"

    # Upstream drift in the last target must leave every earlier target unwritten too.
    pristine = snapshot(root)
    last = TARGETS[-1]
    block = PATCHER_MODULE.BLOCKS[last][-1]
    drifted = pristine[last].decode("utf-8").replace(block.original, block.original.replace("\n", " \n", 1)).encode("utf-8")
    (root / last).write_bytes(drifted)
    result = run_patcher(root, check=False)
    assert result.returncode != 0 and f"MultiDiffusion source for {block.name}" in result.stderr
    assert snapshot(root) == {**pristine, last: drifted}
    (root / last).write_bytes(pristine[last])

    tilevae.write_bytes(pristine["scripts/tilevae.py"].replace(b"\n", b"\r\n"))
    result = run_patcher(root, check=False)
    assert result.returncode != 0 and "CRLF" in result.stderr
    assert snapshot(root) == {**pristine, "scripts/tilevae.py": pristine["scripts/tilevae.py"].replace(b"\n", b"\r\n")}

    # A stray terminal-origin helper next to the upstream split_bboxes would leave two helpers after patching.
    tilevae.write_bytes(pristine["scripts/tilevae.py"])
    (root / UTILS).write_bytes(f"{TERMINAL_HELPER}():\n    pass\n\n".encode("utf-8") + pristine[UTILS])
    result = run_patcher(root, check=False)
    assert result.returncode != 0 and "sentinel x1" in result.stderr
    (root / UTILS).write_bytes(pristine[UTILS])

    run_patcher(root)
    block = PATCHER_MODULE.BLOCKS["scripts/tilevae.py"][0]
    partial = tilevae.read_text(encoding="utf-8").replace(block.patched, block.original)
    tilevae.write_text(partial, encoding="utf-8")
    for extra in ((), ("--check",)):
        result = run_patcher(root, *extra, check=False)
        assert result.returncode != 0 and "partially patched" in result.stderr
    assert tilevae.read_text(encoding="utf-8") == partial


def test_write_failing_midway_leaves_every_target_intact(tmp_path: Path):
    """Integration case of patchlib's write-failure test: a multi-file run that fails while writing changes nothing."""
    root = copy_multidiffusion(tmp_path / "md")
    pristine = snapshot(root)

    def limit_file_size():  # a write past 64 bytes fails with EFBIG partway through, as on a full disk
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (64, 64))

    result = subprocess.run([sys.executable, str(PATCHER), str(root)], capture_output=True, text=True, preexec_fn=limit_file_size)

    assert result.returncode != 0 and "File too large" in result.stderr
    assert snapshot(root) == pristine
    assert not list(root.rglob("*.gb10-tmp"))


def test_patched_text_that_is_not_valid_python_is_not_written(tmp_path: Path):
    root = copy_multidiffusion(tmp_path / "md")
    last = root / TARGETS[-1]
    last.write_bytes(last.read_bytes() + b"\ndef broken(:\n")
    before = snapshot(root)

    result = run_patcher(root, check=False)

    assert result.returncode != 0 and "verification failed (invalid Python)" in result.stderr
    assert snapshot(root) == before


# ---------------------------------------------------------------- terminal tile origins (tile_utils/utils.py)


def multidiffusion_origins(module_path: Path, extent: int, tile: int, overlap: int) -> list[int]:
    source = module_path.read_text(encoding="utf-8")
    if TERMINAL_HELPER in source:
        match = re.search(
            r"def _gb10_terminal_tile_origins\(extent:int, tile:int, overlap:int\).*?(?=\n\ndef split_bboxes)",
            source,
            flags=re.S,
        )
        assert match, "patched terminal-origin helper not found"
        namespace: dict[str, object] = {}
        exec("from typing import List\n" + match.group(0), namespace)
        return namespace["_gb10_terminal_tile_origins"](extent, tile, overlap)  # type: ignore[index,operator]

    match = re.search(
        r"cols = math\.ceil\(\(w - overlap\) / \(tile_w - overlap\)\).*?x = min\(int\(col \* dx\), w - tile_w\)",
        source,
        flags=re.S,
    )
    assert match, "upstream MultiDiffusion origin calculation not found"
    cols = math.ceil((extent - overlap) / (tile - overlap))
    dx = (extent - tile) / (cols - 1) if cols > 1 else 0
    return [min(int(col * dx), extent - tile) for col in range(cols)]


def assert_axis_coverage(origins: list[int], extent: int, tile: int) -> None:
    assert origins == sorted(origins)
    assert len(origins) == len(set(origins))
    assert all(origin >= 0 for origin in origins)
    assert origins[0] == 0
    assert origins[-1] == max(0, extent - tile)
    covered = [False] * extent
    for origin in origins:
        for idx in range(origin, min(origin + tile, extent)):
            covered[idx] = True
    assert all(covered)


def test_upstream_tile_origins_still_have_the_proven_floor_rounding_gap(tmp_path: Path):
    utils = copy_multidiffusion(tmp_path / "md") / UTILS
    origins = multidiffusion_origins(utils, extent=555, tile=64, overlap=16)

    assert origins[-3:] == [401, 446, 490]
    assert 491 not in origins
    assert origins[-1] != 555 - 64


def test_patched_tile_origins_cover_the_terminal_edges(tmp_path: Path):
    root = copy_multidiffusion(tmp_path / "md")
    utils = root / UTILS
    assert multidiffusion_origins(utils, extent=160, tile=64, overlap=16) == [0, 48, 96]
    run_patcher(root)

    x_origins = multidiffusion_origins(utils, extent=555, tile=64, overlap=16)
    y_origins = multidiffusion_origins(utils, extent=427, tile=64, overlap=16)
    assert x_origins[-2:] == [480, 491]
    assert y_origins[-2:] == [336, 363]
    assert (x_origins[0], y_origins[0]) == (0, 0)
    assert (x_origins[-1], y_origins[-1]) == (491, 363)
    assert_axis_coverage(x_origins, 555, 64)
    assert_axis_coverage(y_origins, 427, 64)
    # A grid that already ends on the terminal origin is unchanged.
    assert multidiffusion_origins(utils, extent=160, tile=64, overlap=16) == [0, 48, 96]


# ---------------------------------------------------------------- differential tests (CPU)


class NansException(Exception):
    pass


def fork_sdpa_helper():
    """The fork's real run_scaled_dot_product_attention (and the backend parsing it uses), without importing webui."""
    source = (ROOT / "modules" / "sd_hijack_optimizations.py").read_text(encoding="utf-8")
    wanted = {
        "_active_sdpa_backend", "_SDPA_BACKEND_ALIASES", "_normalize_sdpa_backend_choice", "_ALL_SDPA_BACKENDS",
        "_sdpa_backend_selection", "_sdpa_kernel_all_backends_is_noop", "run_scaled_dot_product_attention",
    }
    nodes = [
        node for node in ast.parse(source).body
        if getattr(node, "name", None) in wanted or any(getattr(target, "id", None) in wanted for target in getattr(node, "targets", []))
    ]
    assert len(nodes) == len(wanted)
    namespace = {"torch": torch, "SDPBackend": SDPBackend, "sdpa_kernel": sdpa_kernel, "functools": functools}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "sd_hijack_optimizations.py", "exec"), namespace)
    return namespace["run_scaled_dot_product_attention"]


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

    fork_run_sdpa = fork_sdpa_helper()

    def run_scaled_dot_product_attention(q, k, v, **kwargs):
        calls["sdpa"].append((q.dim(), kwargs.get("sdpa_backend_override")))
        calls.setdefault("sdpa_dtypes", []).append((q.dtype, torch.is_autocast_enabled("cpu")))
        return fork_run_sdpa(q, k, v, **kwargs)

    state = types.SimpleNamespace(interrupted=False, sampling_step=0, sampling_steps=4)
    sd_model = types.SimpleNamespace(cond_stage_key="txt", model=types.SimpleNamespace(conditioning_key="crossattn"))
    shared = types.SimpleNamespace(state=state, opts=types.SimpleNamespace(upcast_attn=False), sd_model=sd_model)

    stub("gradio")
    stub("gradio.components", Component=object)
    stub("modules", __path__=[])
    stub("modules.scripts", Script=object, AlwaysVisible=object())
    def without_autocast(disable=False):
        # modules.devices.without_autocast, for the CPU autocast these tests can run under.
        return torch.autocast("cpu", enabled=False) if torch.is_autocast_enabled("cpu") and not disable else contextlib.nullcontext()

    stub("modules.devices", device=cpu, cpu=cpu, get_optimal_device=lambda: cpu, get_optimal_device_name=lambda: "cpu",
         torch_gc=torch_gc, autocast=contextlib.nullcontext, test_for_nans=test_for_nans, NansException=NansException,
         without_autocast=without_autocast)
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
    """(original, patched) checkouts. The original keeps the terminal tile origins patched, so every differential
    isolates the other changes; the origin change itself is intentionally not bit-identical (see above)."""
    original = copy_multidiffusion(tmp_path / "original")
    apply_block(original, UTILS, "MD terminal tile origins")
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


@pytest.fixture()
def attn_pair(md_pair, import_extension):
    original, patched = md_pair
    (old,) = import_extension(original, "tile_utils.attn")
    (new,) = import_extension(patched, "tile_utils.attn")
    return old, new


def attn_block(channels, dtype=torch.float32):
    from sgm.modules.diffusionmodules.model import AttnBlock

    torch.manual_seed(4)
    return AttnBlock(channels).eval().to(dtype)


def reference_attention(block, h):
    """Independent float64 oracle: proj_out(softmax(q k^T / sqrt(c)) v), without the residual (Tiled VAE adds it)."""
    block = copy.deepcopy(block).double()
    x = h.double()
    q, k, v = block.q(x), block.k(x), block.v(x)
    b, c, height, width = q.shape
    q, k, v = (t.reshape(b, c, height * width) for t in (q, k, v))
    scores = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
    return block.proj_out((v @ scores.transpose(1, 2)).reshape(b, c, height, width))


@pytest.mark.parametrize("forward,backend_override", [("sdp_attnblock_forward", None), ("sdp_no_mem_attnblock_forward", "flash,math")])
@pytest.mark.parametrize("channels_last", [False, True])
def test_tiled_vae_sdp_attention_is_4d_and_numerically_equivalent(attn_pair, webui_stubs, forward, backend_override, channels_last):
    old, new = attn_pair
    block = attn_block(64)
    torch.manual_seed(5)
    h = torch.randn((2, 64, 9, 11))
    if channels_last:
        block = block.to(memory_format=torch.channels_last)
        h = h.to(memory_format=torch.channels_last)

    with torch.no_grad():
        expected = reference_attention(block, h)
        before = getattr(old, forward)(block, h)
        assert webui_stubs.calls["sdpa"] == []
        after = getattr(new, forward)(block, h)

    assert webui_stubs.calls["sdpa"] == [(4, backend_override)]
    assert after.shape == before.shape == h.shape and after.dtype == before.dtype
    torch.testing.assert_close(before.double(), expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(after.double(), expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)


def test_tiled_vae_sdp_attention_upcast_keeps_dtype(attn_pair, webui_stubs):
    old, new = attn_pair
    webui_stubs.shared.opts.upcast_attn = True
    block = attn_block(64, torch.bfloat16)
    torch.manual_seed(6)
    h = torch.randn((1, 64, 8, 8), dtype=torch.bfloat16)
    with torch.no_grad():
        expected = reference_attention(block, h)
        before = old.sdp_attnblock_forward(block, h)
        after = new.sdp_attnblock_forward(block, h)
    assert before.dtype == after.dtype == torch.bfloat16
    torch.testing.assert_close(after.double(), expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(after, before, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("forward", ["sdp_attnblock_forward", "sdp_no_mem_attnblock_forward"])
def test_tiled_vae_sdp_attention_upcast_runs_without_autocast(attn_pair, webui_stubs, forward):
    """upcast_attn must reach the kernel as float32: autocast lists SDPA as lower precision and would cast it back."""
    _old, new = attn_pair
    block = attn_block(64)
    torch.manual_seed(7)
    h = torch.randn((1, 64, 8, 8))
    for upcast in (True, False):
        webui_stubs.shared.opts.upcast_attn = upcast
        webui_stubs.calls["sdpa_dtypes"] = []
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = getattr(new, forward)(block, h)
        assert webui_stubs.calls["sdpa_dtypes"] == [(torch.float32, False) if upcast else (torch.bfloat16, True)]
        assert out.dtype == torch.bfloat16
    with torch.no_grad():
        expected = reference_attention(block, h)
    webui_stubs.shared.opts.upcast_attn = True
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(getattr(new, forward)(block, h).double(), expected, rtol=3e-2, atol=3e-2)


def test_tiled_vae_decode_with_sdp_attention_matches(md_pair, import_extension, webui_stubs):
    original, patched = md_pair
    webui_stubs.hijack.model_hijack.optimization_method = "sdp"
    (old,) = import_extension(original, "scripts/tilevae.py")
    (new,) = import_extension(patched, "scripts/tilevae.py")
    _encoder, decoder = tiny_vae()
    torch.manual_seed(7)
    z = torch.randn((1, 4, 40, 48))
    expected = run_hook(old, decoder, z.clone(), is_decoder=True, tile_size=12, fast=False)
    assert webui_stubs.calls["sdpa"] == []
    actual = run_hook(new, decoder, z.clone(), is_decoder=True, tile_size=12, fast=False)
    # One mid-block attention per tile (2 x 3 tiles), all through the 4-D helper.
    assert webui_stubs.calls["sdpa"] == [(4, None)] * 6
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def fake_unet(x, t, cond):
    """Deterministic stand-in for apply_model: depends on every per-tile input the delegate assembles, and on the
    whole tile (its mean), so overlapping tiles disagree and the blend weights matter."""
    text = cond["crossattn"].mean(dim=(1, 2)).view(-1, 1, 1, 1)
    vector = cond["vector"].mean(dim=1).view(-1, 1, 1, 1)
    tile_mean = x.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1)
    return torch.tanh(x * 1.3 + cond["c_concat"][:, :4] * 0.1 + text + vector * 0.5 + tile_mean * 2.0 + t.view(-1, 1, 1, 1) * 0.01)


def make_mixture_of_diffusers(module, *, width, height, tile, overlap, tile_batch_size):
    p = types.SimpleNamespace(sampler_name="Euler a", width=width, height=height, disable_extra_networks=True)
    delegate = module.MixtureOfDiffusers(p, module.KDiffusionSampler())
    delegate.init_grid_bbox(tile, tile, overlap, tile_batch_size)
    delegate.init_done()
    return delegate


@pytest.mark.parametrize("width,height,tile,overlap,tile_batch_size", [
    (1024, 768, 64, 16, 4),
    (1000, 744, 48, 8, 3),   # latent 125x93: terminal tiles off the stride grid
    (512, 512, 96, 8, 4),    # one tile clamped to the latent size
])
def test_mixture_of_diffusers_precomputed_tile_weights_are_bit_identical(md_pair, import_extension, webui_stubs, width, height, tile, overlap, tile_batch_size):
    original, patched = md_pair
    webui_stubs.shared.sd_model.apply_model_original_md = fake_unet
    (old,) = import_extension(original, "tile_methods.mixtureofdiffusers")
    (new,) = import_extension(patched, "tile_methods.mixtureofdiffusers")
    before = make_mixture_of_diffusers(old, width=width, height=height, tile=tile, overlap=overlap, tile_batch_size=tile_batch_size)
    after = make_mixture_of_diffusers(new, width=width, height=height, tile=tile, overlap=overlap, tile_batch_size=tile_batch_size)
    assert [len(weights) for weights in after.batched_tile_weights] == [len(bboxes) for bboxes in after.batched_bboxes]

    torch.manual_seed(8)
    n, h, w = 2, height // 8, width // 8
    cond = {"crossattn": torch.randn((n, 77, 32)), "vector": torch.randn((n, 16)), "c_concat": torch.randn((n, 5, h, w))}
    for step in range(3):
        webui_stubs.state.sampling_step = step
        x = torch.randn((n, 4, h, w))
        sigma = torch.full((n,), 10.0 / (step + 1))
        expected = before.apply_model_hijack(x, sigma, dict(cond)).clone()
        actual = after.apply_model_hijack(x, sigma, dict(cond)).clone()
        assert after.x_buffer is not None  # the tiled path ran, not the full-size fallback
        assert torch.equal(actual, expected)


def test_mixture_of_diffusers_without_grid_tiles_precomputes_nothing(md_pair, import_extension, webui_stubs):
    _original, patched = md_pair
    (new,) = import_extension(patched, "tile_methods.mixtureofdiffusers")
    p = types.SimpleNamespace(sampler_name="Euler a", width=512, height=512, disable_extra_networks=True)
    delegate = new.MixtureOfDiffusers(p, new.KDiffusionSampler())
    # Region control without background drawing: no grid bboxes and no tile_weights attribute at all.
    delegate.enable_custom_bbox = True
    delegate.draw_background = False
    delegate.custom_bboxes = [types.SimpleNamespace(blend_mode=new.BlendMode.FOREGROUND)]
    delegate.init_done()
    assert delegate.batched_tile_weights == []
    assert not hasattr(delegate, "tile_weights")
