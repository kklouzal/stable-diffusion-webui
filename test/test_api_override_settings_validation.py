import ast
import types
from pathlib import Path

import pytest
from fastapi.exceptions import HTTPException

API_PATH = Path(__file__).resolve().parents[1] / "modules" / "api" / "api.py"
OPTIONS_PATH = Path(__file__).resolve().parents[1] / "modules" / "options.py"


def load_function(path, name, namespace):
    """Compile one top-level function from a module without importing the webui runtime."""
    tree = ast.parse(path.read_text(encoding="utf8"))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def load_options_methods(*names):
    """Options methods compiled as plain functions (self passed explicitly)."""
    tree = ast.parse(OPTIONS_PATH.read_text(encoding="utf8"))
    options_cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Options")
    body = [node for node in options_cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(OPTIONS_PATH), "exec"), namespace)
    return [namespace[name] for name in names]


@pytest.fixture
def validate():
    return load_function(API_PATH, "_validate_override_settings", {
        "HTTPException": HTTPException,
        "sd_models": types.SimpleNamespace(checkpoint_aliases={"real.safetensors": object(), "real": object()}),
        "shared_items": types.SimpleNamespace(sd_vae_items=lambda: ["Automatic", "None", "sdxl_vae.safetensors"]),
    })


@pytest.fixture
def opts():
    same_type, api_type_mismatch = load_options_methods("same_type", "api_type_mismatch")
    labels = {
        "CLIP_stop_at_last_layers": types.SimpleNamespace(default=1),
        "eta_noise_seed_delta": types.SimpleNamespace(default=0),
        "upcast_attn": types.SimpleNamespace(default=False),
        "sd_vae": types.SimpleNamespace(default="Automatic"),
        "sd_model_checkpoint": types.SimpleNamespace(default=None),
        "s_noise": types.SimpleNamespace(default=1.0),
    }
    namespace = types.SimpleNamespace(data={"legacy_key": 5}, data_labels=labels, typemap={int: float})
    namespace.same_type = lambda x, y: same_type(namespace, x, y)
    namespace.api_type_mismatch = lambda key, value: api_type_mismatch(namespace, key, value)
    return namespace


def test_values_of_the_option_type_pass(validate, opts):
    validate({"CLIP_stop_at_last_layers": 2, "upcast_attn": True, "sd_vae": "None", "s_noise": 1, "eta_noise_seed_delta": None}, opts)
    validate(None, opts)


@pytest.mark.parametrize("override", [{"upcast_attn": "false"}, {"CLIP_stop_at_last_layers": "2"}, {"sd_vae": 3}])
def test_a_value_of_another_type_is_a_422(validate, opts, override):
    with pytest.raises(HTTPException) as exc:
        validate(override, opts)
    assert exc.value.status_code == 422
    assert next(iter(override)) in exc.value.detail


def test_type_error_detail_text_is_unchanged(validate, opts):
    with pytest.raises(HTTPException) as exc:
        validate({"upcast_attn": "false"}, opts)
    assert exc.value.detail == "override_settings: option 'upcast_attn' expects a value of type bool, got str 'false'"


def test_unknown_option_is_a_422_unless_unchanged(validate, opts):
    with pytest.raises(HTTPException) as exc:
        validate({"no_such_option": 1}, opts)
    assert exc.value.status_code == 422
    validate({"legacy_key": 5}, opts)  # equal to the stored value: opts.set would leave it alone


def test_available_checkpoint_and_vae_pass(validate, opts):
    validate({"sd_model_checkpoint": "real", "sd_vae": "sdxl_vae.safetensors"}, opts)
    validate({"sd_model_checkpoint": "real.safetensors", "sd_vae": "Automatic"}, opts)


@pytest.mark.parametrize("override", [{"sd_model_checkpoint": "typo.safetensors"}, {"sd_model_checkpoint": None},
                                      {"sd_model_checkpoint": ["real"]}, {"sd_vae": "typo.safetensors"}, {"sd_vae": None}])
def test_unavailable_checkpoint_or_vae_is_a_422(validate, opts, override):
    # The generation replaced them silently (the configured checkpoint, no VAE).
    with pytest.raises(HTTPException) as exc:
        validate(override, opts)
    assert exc.value.status_code == 422
    assert next(iter(override)) in exc.value.detail
