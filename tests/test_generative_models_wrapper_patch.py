import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

PATCH = Path("patches/generative-models/0011-openai-wrapper-skip-empty-concat.patch").resolve()
WRAPPERS = Path("sgm/modules/diffusionmodules/wrappers.py")
BAKED = Path("repositories/generative-models")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def old_and_new_wrappers(tmp_path):
    if not (BAKED / WRAPPERS).exists():
        pytest.skip("needs the image's baked generative-models checkout")
    trees = {}
    for name in ("old", "new"):
        (tmp_path / name / WRAPPERS).parent.mkdir(parents=True)
        shutil.copy(BAKED / WRAPPERS, tmp_path / name / WRAPPERS)
        trees[name] = tmp_path / name
    # The baked copy is pre- or post-patch depending on the image; derive the other side with git apply,
    # using docker/apply-local-patches.py's flags.
    apply = ["git", "apply", "--ignore-whitespace"]
    already_applied = subprocess.run([*apply, "--reverse", "--check", str(PATCH)], cwd=trees["old"]).returncode == 0
    if already_applied:
        subprocess.run([*apply, "--reverse", str(PATCH)], cwd=trees["old"], check=True)
    else:
        subprocess.run([*apply, "--check", str(PATCH)], cwd=trees["new"], check=True)
        subprocess.run([*apply, str(PATCH)], cwd=trees["new"], check=True)
    return _load(trees["old"] / WRAPPERS, "wrappers_old"), _load(trees["new"] / WRAPPERS, "wrappers_new")


class _Recorder(torch.nn.Module):
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        self.seen = x
        return x * 2 + timesteps.reshape(-1, 1, 1, 1) + context.mean() + y.mean()


@pytest.mark.parametrize("memory_format", [torch.contiguous_format, torch.channels_last])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_concat", [False, True])
def test_patched_wrapper_feeds_the_unet_the_same_tensor(old_and_new_wrappers, memory_format, dtype, with_concat):
    old, new = old_and_new_wrappers
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 4, 16, 16, generator=generator).to(dtype).to(memory_format=memory_format)
    c = {"crossattn": torch.randn(2, 77, 32, generator=generator), "vector": torch.randn(2, 16, generator=generator)}
    if with_concat:
        c["concat"] = torch.randn(2, 5, 16, 16, generator=generator).to(dtype)
    t = torch.tensor([500.0, 500.0])

    old_unet, new_unet = _Recorder(), _Recorder()
    out_old = old.OpenAIWrapper(old_unet)(x, t, c)
    out_new = new.OpenAIWrapper(new_unet)(x, t, c)

    assert torch.equal(out_old, out_new)
    assert torch.equal(old_unet.seen, new_unet.seen)
    assert (new_unet.seen.dtype, new_unet.seen.stride()) == (old_unet.seen.dtype, old_unet.seen.stride())
