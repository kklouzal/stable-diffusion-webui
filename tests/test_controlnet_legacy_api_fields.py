from pathlib import Path
import subprocess
import sys


WEBUI_ROOT = Path(__file__).resolve().parents[1]
MODELS_SOURCE = WEBUI_ROOT / "modules" / "api" / "models.py"
API_SOURCE = WEBUI_ROOT / "modules" / "api" / "api.py"
CONTROLNET_ROOT = WEBUI_ROOT / "extensions" / "sd-webui-controlnet"
CONTROLNET_SOURCE = CONTROLNET_ROOT / "scripts" / "controlnet.py"


def test_controlnet_legacy_remote_fields_are_explicit_api_model_fields():
    source = MODELS_SOURCE.read_text()

    expected_bases = [
        "enabled",
        "module",
        "model",
        "weight",
        "image",
        "resize_mode",
        "lowvram",
        "pres",
        "pthr_a",
        "pthr_b",
        "guidance_start",
        "guidance_end",
        "control_mode",
        "pixel_perfect",
    ]
    expected_aliases = [
        "input_image",
        "low_vram",
        "processor_res",
        "threshold_a",
        "threshold_b",
        "guidance_strength",
    ]

    for base in expected_bases:
        assert f'"{base}":' in source
    for alias in expected_aliases:
        assert f'"{alias}":' in source

    assert 'for suffix in ("", "2", "3")' in source
    assert 'f"control_net_{name}{suffix}"' in source
    assert 'name != "image"' not in source
    assert source.count("*control_net_api_fields(),") == 2


def test_generated_api_models_support_protected_pydantic_v2():
    source = MODELS_SOURCE.read_text()

    assert "ConfigDict(populate_by_name=True, frozen=False)" in source
    assert "current_image: Optional[str] = Field(default=None" in source
    assert "textinfo: Optional[str] = Field(default=None" in source


def test_controlnet_unit_uses_native_pydantic_v2_validators():
    # ControlNet's own ControlNetUnit tests are the behavioral oracle for the v1 -> v2 validator migration; they need
    # the extension root as the import root. Any leftover v1 API (@validator, @root_validator, class Config, .copy())
    # emits PydanticDeprecatedSince20, which is an error here.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-W", "error::pydantic.warnings.PydanticDeprecatedSince20", "unit_tests"],
        cwd=CONTROLNET_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


def test_controlnet_legacy_aliases_are_normalized_before_processing():
    source = API_SOURCE.read_text()

    assert '"input_image": "image"' in source
    assert '"low_vram": "lowvram"' in source
    assert '"processor_res": "pres"' in source
    assert '"threshold_a": "pthr_a"' in source
    assert '"threshold_b": "pthr_b"' in source
    assert '"guidance_strength": "guidance_end"' in source
    assert '_normalize_controlnet_remote_aliases(args)' in source
    assert '_pop_controlnet_remote_args(args)' in source
    assert '_attach_controlnet_remote_args(p, controlnet_remote_args)' in source
    assert 'value = args.pop(key)' in source


def test_controlnet_extension_reads_indexed_legacy_remote_units():
    source = CONTROLNET_SOURCE.read_text()

    assert 'getattr(p, f"{attribute}{idx + 1}", None)' in source
    assert 'for idx in range(external_code.get_max_models_num())' in source
    assert 'normalize_remote_resize_mode' in source
    assert 'normalize_remote_control_mode' in source
    assert 'normalize_guidance_interval' in source
