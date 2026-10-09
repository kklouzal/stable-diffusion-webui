"""Face restoration: restored faces are quantized like the reference GFPGAN/CodeFormer pipeline (basicsr
`tensor2img`, which rounds), and a restorer whose model cannot be loaded or whose inference fails fails the request
instead of silently returning the unrestored image."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def fr_utils():
    pkg = types.ModuleType("modules")
    pkg.__path__ = []
    shared = types.ModuleType("modules.shared")
    shared.opts = SimpleNamespace(face_restoration_unload=False, face_restoration_model=None)
    shared.face_restorers = []
    devices = types.ModuleType("modules.devices")
    devices.torch_gc = lambda: None
    devices.cpu = torch.device("cpu")
    stubs = {"modules": pkg, "modules.shared": shared, "modules.devices": devices,
             "modules.face_restoration": None, "modules.face_restoration_utils": None}
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        for name in ("shared", "devices"):
            setattr(pkg, name, stubs[f"modules.{name}"])
        for name in ("face_restoration", "face_restoration_utils"):
            spec = importlib.util.spec_from_file_location(f"modules.{name}", ROOT / "modules" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"modules.{name}"] = module
            spec.loader.exec_module(module)
            setattr(pkg, name, module)
        yield pkg.face_restoration_utils
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _FakeFaceHelper:
    def __init__(self, faces):
        self.cropped_faces = faces
        self.restored = []

    def clean_all(self):
        self.cleaned = getattr(self, "cleaned", 0) + 1

    def read_image(self, img):
        self.input_img = np.ascontiguousarray(img)

    def get_face_landmarks_5(self, **kwargs):
        pass

    def align_warp_face(self):
        pass

    def add_restored_face(self, face):
        self.restored.append(face)

    def get_inverse_affine(self, _save_path):
        pass

    def paste_faces_to_input_image(self):
        return self.input_img


def _basicsr_tensor2img(tensor, min_max=(-1, 1)):
    """basicsr.utils.img_util.tensor2img(tensor, rgb2bgr=True, min_max=min_max) for a 1x3xHxW tensor."""
    t = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
    t = (t - min_max[0]) / (min_max[1] - min_max[0])
    img = cv2.cvtColor(t.numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    return (img * 255.0).round().astype(np.uint8)


def test_restored_face_is_rounded_like_the_reference(fr_utils):
    rng = np.random.default_rng(0)
    restored = torch.from_numpy(rng.uniform(-1.1, 1.1, size=(1, 3, 32, 32)).astype(np.float32))
    helper = _FakeFaceHelper([rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)])
    image = rng.integers(0, 256, size=(40, 48, 3), dtype=np.uint8)

    fr_utils.restore_with_face_helper(image, helper, lambda face: restored.clone(), torch.device("cpu"))

    expected = _basicsr_tensor2img(restored)
    assert helper.restored[0].dtype == np.uint8
    assert np.array_equal(helper.restored[0], expected)
    unit = ((restored.squeeze(0).clamp(-1, 1) + 1) / 2).numpy().transpose(1, 2, 0)
    truncated = (cv2.cvtColor(unit, cv2.COLOR_RGB2BGR) * 255.0).astype(np.uint8)
    assert not np.array_equal(truncated, expected)  # the former truncation is ~0.5 LSB darker on average


def test_unloadable_restorer_fails_instead_of_returning_unrestored(fr_utils, tmp_path):
    class Broken(fr_utils.CommonFaceRestoration):
        def name(self):
            return "Broken"

        def get_device(self):
            return torch.device("cpu")

        def load_net(self):
            raise OSError("missing weights")

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Unable to load Broken face-restoration model: missing weights"):
        Broken(str(tmp_path)).restore_with_helper(image, lambda face: face)


def test_failed_face_inference_fails_instead_of_pasting_the_unrestored_crop(fr_utils):
    helper = _FakeFaceHelper([np.zeros((8, 8, 3), dtype=np.uint8)])

    def broken(face):
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        fr_utils.restore_with_face_helper(np.zeros((8, 8, 3), dtype=np.uint8), helper, broken, torch.device("cpu"))
    assert helper.restored == []
    assert helper.cleaned == 2  # before detection and in the finally block
