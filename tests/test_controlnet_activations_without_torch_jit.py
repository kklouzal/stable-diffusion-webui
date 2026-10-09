import importlib.util
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

ANNOTATOR = Path(__file__).resolve().parents[1] / "extensions" / "sd-webui-controlnet" / "annotator"
GEFFNET_ACTIVATIONS = ANNOTATOR / "normalbae" / "models" / "submodules" / "efficientnet_repo" / "geffnet" / "activations"


def _load(path):
    # torch 2.14 deprecates torch.jit.script and warns when the decorator runs, i.e. while the module executes.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = importlib.util.spec_from_file_location(f"_controlnet_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    assert not [w for w in caught if "torch.jit" in str(w.message)], path
    return module


def test_teed_activations_are_eager():
    x = torch.linspace(-8, 8, 2001, dtype=torch.float32)
    smish = _load(ANNOTATOR / "teed" / "Fsmish.py").smish

    assert torch.equal(smish(x), x * torch.tanh(torch.log(1 + torch.sigmoid(x))))


def test_geffnet_activations_are_eager_and_hand_written_backward_matches_autograd():
    jit = _load(GEFFNET_ACTIVATIONS / "activations_jit.py")
    me = _load(GEFFNET_ACTIVATIONS / "activations_me.py")
    references = {
        "swish": lambda x: x * torch.sigmoid(x),
        "mish": lambda x: x * torch.tanh(F.softplus(x)),
        "hard_sigmoid": lambda x: F.relu6(x + 3) / 6,
        "hard_swish": lambda x: x * F.relu6(x + 3) / 6,
    }
    # The offset keeps samples off the hard-* kinks at +-3, where hand-written and autograd subgradients may differ.
    x = torch.linspace(-8, 8, 1600, dtype=torch.float64) + 1e-3

    for name, reference in references.items():
        expected = reference(x)
        torch.testing.assert_close(getattr(jit, f"{name}_jit")(x), expected)

        x_reference = x.clone().requires_grad_()
        reference(x_reference).sum().backward()
        x_me = x.clone().requires_grad_()
        out = getattr(me, f"{name}_me")(x_me)
        out.sum().backward()
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(x_me.grad, x_reference.grad)
