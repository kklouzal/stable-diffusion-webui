from pathlib import Path
import subprocess
import sys


WEBUI_ROOT = Path(__file__).resolve().parents[1]
MODELS_SOURCE = WEBUI_ROOT / "modules" / "api" / "models.py"
API_SOURCE = WEBUI_ROOT / "modules" / "api" / "api.py"
CONTROLNET_SOURCE = WEBUI_ROOT / "extensions" / "sd-webui-controlnet" / "scripts" / "controlnet.py"


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
    assert 'hasattr(BaseModel, "model_fields")' in source
    assert "current_image: Optional[str] = Field(default=None" in source
    assert "textinfo: Optional[str] = Field(default=None" in source


def test_controlnet_root_validators_are_patched_for_pydantic_v2(tmp_path: Path):
    args_module = tmp_path / "args.py"
    args_module.write_text(
        "class ControlNetUnit:\n"
        "    @root_validator\n"
        "    def bound_check_params(cls, values):\n"
        "        return values\n"
        "    @root_validator\n"
        "    def guidance_check(cls, values):\n"
        "        return values\n"
    )
    patcher = WEBUI_ROOT / "gb10" / "patch-controlnet-pydantic2.py"

    subprocess.run([sys.executable, str(patcher), str(args_module)], check=True)
    subprocess.run([sys.executable, str(patcher), "--check", str(args_module)], check=True)

    assert args_module.read_text().count("@root_validator(skip_on_failure=True)") == 2


def test_launcher_applies_controlnet_pydantic2_patch():
    launcher = (WEBUI_ROOT / "gb10" / "run.sh").read_text()

    assert launcher.count('patch-controlnet-pydantic2.py"') == 2


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
