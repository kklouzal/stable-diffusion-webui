"""modules.sd_vae_taesd: decoder_model()/encoder_model() load the published per-family TAESD weights (whose keys
are those of the bare decoder()/encoder() Sequential) into the right network, downloading each file once and
caching the loaded network."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from test.helpers import load_source, module, stub_modules

torch = pytest.importorskip("torch")


@pytest.fixture()
def taesd(tmp_path):
    # cmd_opts: read by modules.safe's torch.load wrapper (once another test has imported the real webui modules),
    # which imports modules.shared when called, so the stubs stay installed for the test.
    shared = module("modules.shared", cmd_opts=SimpleNamespace(disable_safe_unpickle=False))
    downloads = []

    def load_file_from_url(url, *, model_dir, file_name):
        downloads.append((url, model_dir, file_name))
        return str(Path(model_dir) / file_name)

    with stub_modules({
        "modules": module("modules", package=True),
        "modules.devices": module("modules.devices", device=torch.device("cpu"), dtype=torch.float32),
        "modules.paths_internal": module("modules.paths_internal", models_path=str(tmp_path)),
        "modules.shared": shared,
        "modules.util": module("modules.util", load_file_from_url=load_file_from_url),
    }):
        yield SimpleNamespace(
            module=load_source("modules.sd_vae_taesd", "modules/sd_vae_taesd.py"), shared=shared, downloads=downloads,
            model_dir=tmp_path / "VAE-taesd",
        )


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
