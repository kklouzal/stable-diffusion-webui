import numpy as np
import cv2
import os
import threading

import torch


def load_model(filename: str, remote_url: str, model_dir: str) -> str:
    """
    Load the model from the specified filename and remote URL if it doesn't exist locally.

    Args:
        filename (str): The filename of the model.
        remote_url (str): The remote URL of the model.
    """
    local_path = os.path.join(model_dir, filename)
    if not os.path.exists(local_path):
        from scripts.utils import load_file_from_url
        load_file_from_url(remote_url, model_dir=model_dir)
    return local_path


_meta_parameters_lock = threading.Lock()


def _build_with_meta_parameters(build):
    """build() with every parameter it registers in this thread moved to the meta device, so no parameter storage is
    allocated or randomly initialized. Buffers are built as usual: non-persistent ones (absent from any state dict)
    keep their computed values. Parameters other threads register meanwhile are not affected."""
    owner = threading.get_ident()
    register_parameter = torch.nn.Module.register_parameter

    def register_meta_parameter(module, name, param):
        register_parameter(module, name, param)
        if param is not None and not param.is_meta and threading.get_ident() == owner:
            module._parameters[name] = torch.nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)

    with _meta_parameters_lock:
        torch.nn.Module.register_parameter = register_meta_parameter
        try:
            return build()
        finally:
            torch.nn.Module.register_parameter = register_parameter


def _as_copied_into(value, target):
    """`value` as copying it into `target` (load_state_dict without assign) leaves it: target's dtype and strides."""
    if value.shape != target.shape or (value.dtype == target.dtype and value.stride() == target.stride()):
        return value  # a shape mismatch is load_state_dict's error to raise
    return torch.empty_strided(target.shape, target.stride(), dtype=target.dtype).copy_(value)


def build_with_state_dict(build, state_dict, unused_keys=frozenset()):
    """The module build() returns, holding `state_dict` exactly as build() + load_state_dict(state_dict) leaves it,
    without allocating and randomly initializing the parameters the checkpoint overwrites.

    The model's tensors take the checkpoint's values with the model's dtypes and strides; checkpoint tensors that
    already match are used as they are (they may be memory-mapped). The checkpoint must hold every key of the
    model's state dict, and its other keys must be exactly `unused_keys`: any other difference raises RuntimeError,
    as does a parameter shared by several modules (assigning would untie it). The module stays on the CPU, in the
    mode build() left it in.
    """
    model = _build_with_meta_parameters(build)
    shared = sorted(dict(model.named_parameters(remove_duplicate=False)).keys() - dict(model.named_parameters()).keys())
    if shared:
        raise RuntimeError(f"{type(model).__name__} shares parameters, which loading by assignment would untie: {shared}")
    expected = model.state_dict()
    missing = sorted(expected.keys() - state_dict.keys())
    unexpected = state_dict.keys() - expected.keys()
    if missing or unexpected != unused_keys:
        raise RuntimeError(
            f"Unsupported {type(model).__name__} checkpoint: missing keys {missing}, "
            f"unexpected keys {sorted(unexpected - unused_keys)}, absent unused keys {sorted(unused_keys - unexpected)}")
    model.load_state_dict(
        {key: _as_copied_into(state_dict[key], target) for key, target in expected.items()}, strict=True, assign=True)
    unloaded = [name for name, tensor in (*model.named_parameters(), *model.named_buffers()) if tensor.is_meta]
    if unloaded:
        raise RuntimeError(f"{type(model).__name__} tensors not loaded from the checkpoint: {unloaded}")
    return model


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def make_noise_disk(H, W, C, F):
    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F: F + H, F: F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


def min_max_norm(x):
    x -= np.min(x)
    x /= np.maximum(np.max(x), 1e-5)
    return x


def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y
