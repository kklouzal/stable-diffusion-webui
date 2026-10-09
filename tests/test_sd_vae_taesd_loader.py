"""modules.sd_vae_taesd: decoder_model()/encoder_model() load the published per-family TAESD weights (whose keys
are those of the bare decoder()/encoder() Sequential) into the right network, downloading each file once and
caching the loaded network."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
_STUBBED = ("modules", "modules.devices", "modules.paths_internal", "modules.shared", "modules.util", "modules.sd_vae_taesd")


@pytest.fixture()
def taesd(tmp_path):
    previous = {name: sys.modules.get(name) for name in _STUBBED}
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    devices = types.ModuleType("modules.devices")
    devices.device = torch.device("cpu")
    devices.dtype = torch.float32
    paths_internal = types.ModuleType("modules.paths_internal")
    paths_internal.models_path = str(tmp_path)
    shared = types.ModuleType("modules.shared")
    # Read by modules.safe's torch.load wrapper once another test has imported the real webui modules.
    shared.cmd_opts = SimpleNamespace(disable_safe_unpickle=False)
    util = types.ModuleType("modules.util")
    downloads = []

    def load_file_from_url(url, *, model_dir, file_name):
        downloads.append((url, model_dir, file_name))
        return str(Path(model_dir) / file_name)

    util.load_file_from_url = load_file_from_url
    modules_pkg.devices, modules_pkg.paths_internal, modules_pkg.shared = devices, paths_internal, shared
    sys.modules.update({
        "modules": modules_pkg,
        "modules.devices": devices,
        "modules.paths_internal": paths_internal,
        "modules.shared": shared,
        "modules.util": util,
    })
    try:
        spec = importlib.util.spec_from_file_location("modules.sd_vae_taesd", ROOT / "modules" / "sd_vae_taesd.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["modules.sd_vae_taesd"] = module
        spec.loader.exec_module(module)
        yield SimpleNamespace(module=module, shared=shared, downloads=downloads, model_dir=tmp_path / "VAE-taesd")
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


@pytest.mark.parametrize("kind", ["decoder", "encoder"])
@pytest.mark.parametrize(("family", "prefix", "latent_channels"), [
    ("sd1", "taesd", 4),
    ("sdxl", "taesdxl", 4),
    ("sd3", "taesd3", 16),
])
def test_taesd_model_loads_the_family_weights_once(taesd, kind, family, prefix, latent_channels):
    torch.manual_seed(0)
    reference = getattr(taesd.module, kind)(latent_channels).eval()
    file_name = f"{prefix}_{kind}.pth"
    taesd.model_dir.mkdir()
    torch.save(reference.state_dict(), taesd.model_dir / file_name)
    taesd.shared.sd_model = SimpleNamespace(is_sd3=family == "sd3", is_sdxl=family == "sdxl")
    load = getattr(taesd.module, f"{kind}_model")

    model = load()

    assert taesd.downloads == [(f"https://github.com/madebyollin/taesd/raw/main/{file_name}", str(taesd.model_dir), file_name)]
    assert not model.training
    x = torch.randn(1, latent_channels if kind == "decoder" else 3, 16, 16)
    with torch.no_grad():
        assert torch.equal(model(x), reference(x))
    assert load() is model
    assert len(taesd.downloads) == 1
