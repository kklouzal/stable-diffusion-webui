from pathlib import Path
import subprocess
import sys


WEBUI_ROOT = Path(__file__).resolve().parents[1]
API_SOURCE = WEBUI_ROOT / "modules" / "api" / "api.py"
CONTROLNET_ROOT = WEBUI_ROOT / "extensions" / "sd-webui-controlnet"
CONTROLNET_SOURCE = CONTROLNET_ROOT / "scripts" / "controlnet.py"

# The legacy top-level ControlNet remote fields (attributes ControlNet's remote-call adapter reads from p) ...
CONTROLNET_REMOTE_FIELDS = (
    "enabled", "module", "model", "weight", "image", "resize_mode", "lowvram", "pres", "pthr_a", "pthr_b",
    "guidance_start", "guidance_end", "control_mode", "pixel_perfect",
)
# ... and the accepted aliases, with the remote field each one fills.
CONTROLNET_REMOTE_ALIASES = {
    "input_image": "image",
    "low_vram": "lowvram",
    "processor_res": "pres",
    "threshold_a": "pthr_a",
    "threshold_b": "pthr_b",
    "guidance_strength": "guidance_end",
}
UNIT_SUFFIXES = ("", "2", "3")


def _exec_nodes(path, nodes, namespace):
    import ast

    module = ast.Module(body=list(nodes), type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_controlnet_legacy_remote_fields_are_explicit_api_model_fields(initialize):
    from modules.api import models

    keys = {f"control_net_{name}{suffix}" for suffix in UNIT_SUFFIXES for name in (*CONTROLNET_REMOTE_FIELDS, *CONTROLNET_REMOTE_ALIASES)}
    values = {key: f"value-of-{key}" for key in ("control_net_image", "control_net_input_image2", "control_net_module3")}
    # Both generation request models carry the fields: the dynamic models drop unknown keys.
    for model in (models.StableDiffusionTxt2ImgProcessingAPI, models.StableDiffusionImg2ImgProcessingAPI):
        fields = {field.alias: (name, field) for name, field in model.model_fields.items()}
        assert {alias for alias in fields if alias.startswith("control_net_")} == keys
        for key in keys:
            name, field = fields[key]
            assert name == key and field.default is None
        request = model.model_validate({**values, "control_net_weight2": None})
        assert {key: getattr(request, key) for key in values} == values
        assert request.control_net_weight2 is None and request.control_net_enabled is None


def test_generated_api_models_populate_by_name_and_stay_mutable(initialize):
    from modules.api import models

    for model in (models.StableDiffusionTxt2ImgProcessingAPI, models.StableDiffusionImg2ImgProcessingAPI):
        assert model.model_config["populate_by_name"] is True
        assert not model.model_config.get("frozen", False)
        request = model.model_validate({"prompt": "a cat"})
        request.prompt = "a dog"  # the API rewrites requests in place (infotext, sampler/scheduler)
        assert request.prompt == "a dog"
    progress = models.ProgressResponse(progress=0.5, eta_relative=1.0, state={})
    assert progress.current_image is None and progress.textinfo is None


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


def test_controlnet_legacy_aliases_are_normalized_before_processing(initialize):
    import ast
    from types import SimpleNamespace
    from typing import Any

    from modules.api import models

    tree = ast.parse(API_SOURCE.read_text(encoding="utf-8"))
    helpers = {"_normalize_controlnet_remote_aliases", "_controlnet_remote_api_keys", "_pop_controlnet_remote_args", "_attach_controlnet_remote_args"}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in helpers
             or isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_CONTROLNET_REMOTE_ALIAS_FIELDS"]
    assert len(nodes) == len(helpers) + 1
    api = SimpleNamespace(**_exec_nodes(API_SOURCE, nodes, {"Any": Any, "models": models}))

    for suffix in UNIT_SUFFIXES:
        for alias, field in CONTROLNET_REMOTE_ALIASES.items():
            args = {f"control_net_{alias}{suffix}": "from-alias"}
            api._normalize_controlnet_remote_aliases(args)
            assert args == {f"control_net_{field}{suffix}": "from-alias"}

    args = {
        "prompt": "p",
        "control_net_threshold_a2": 0.25, "control_net_pthr_a2": 0.75,  # an explicit remote field wins over its alias
        "control_net_low_vram3": True, "control_net_lowvram3": None,
        "control_net_weight": None, "control_net_module": "canny",
    }
    api._normalize_controlnet_remote_aliases(args)
    assert args == {"prompt": "p", "control_net_pthr_a2": 0.75, "control_net_lowvram3": True,
                    "control_net_weight": None, "control_net_module": "canny"}
    remote = api._pop_controlnet_remote_args(args)
    assert args == {"prompt": "p"}  # nothing ControlNet reaches the processing constructor; unset (None) fields are dropped
    assert remote == {"control_net_pthr_a2": 0.75, "control_net_lowvram3": True, "control_net_module": "canny"}
    p = SimpleNamespace()
    api._attach_controlnet_remote_args(p, remote)
    assert vars(p) == remote

    # Wiring: the request is normalized when it becomes args; the remote args leave args before p is built and are
    # attached to it afterwards.
    api_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    methods = {node.name: node for node in api_class.body if isinstance(node, ast.FunctionDef)}

    def call_lines(method, name):
        return [node.lineno for node in ast.walk(methods[method]) if isinstance(node, ast.Call) and ast.unparse(node.func) == name]

    assert len(call_lines("_prepare_generation_api_request", "_normalize_controlnet_remote_aliases")) == 1
    (pop,) = call_lines("_run_generation_task", "_pop_controlnet_remote_args")
    (build,) = call_lines("_run_generation_task", "processing_class")
    (attach,) = call_lines("_run_generation_task", "_attach_controlnet_remote_args")
    assert pop < build < attach


def _controlnet_remote_reader(allow_script_control):
    """Script.get_enabled_units and the remote-call readers it uses, from the real scripts/controlnet.py."""
    import ast
    import copy
    import importlib.util
    from types import SimpleNamespace
    from typing import List

    spec = importlib.util.spec_from_file_location("controlnet_enums_under_test", CONTROLNET_ROOT / "scripts" / "enums.py")
    enums = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enums)

    class ControlNetUnit:
        """The ControlNetUnit fields and defaults the remote-call reader touches."""

        def __init__(self):
            self.enabled, self.module, self.model, self.weight, self.image = False, "none", "None", 1.0, None
            self.resize_mode, self.control_mode = enums.ResizeMode.INNER_FIT, enums.ControlMode.BALANCED
            self.low_vram, self.processor_res, self.threshold_a, self.threshold_b = False, -1, -1, -1
            self.guidance_start, self.guidance_end, self.pixel_perfect = 0.0, 1.0, False
            self.input_mode, self.accepts_multiple_inputs = enums.InputMode.SIMPLE, False

        def model_copy(self):
            return copy.copy(self)

    wanted = {"normalize_remote_resize_mode", "normalize_remote_control_mode", "normalize_guidance_interval",
              "get_remote_call", "parse_remote_call", "get_enabled_units"}
    tree = ast.parse(CONTROLNET_SOURCE.read_text(encoding="utf-8"))
    script = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Script")
    methods = [node for node in script.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in methods} == wanted
    written = []
    namespace = {
        "ResizeMode": enums.ResizeMode, "ControlMode": enums.ControlMode, "InputMode": enums.InputMode,
        "ControlNetUnit": ControlNetUnit, "List": List,
        "shared": SimpleNamespace(opts=SimpleNamespace(data={"control_net_allow_script_control": allow_script_control})),
        "external_code": SimpleNamespace(get_all_units_in_processing=lambda p: [], get_max_models_num=lambda: 3),
        "Infotext": SimpleNamespace(write_infotext=lambda units, p: written.append(units)),
    }
    _exec_nodes(CONTROLNET_SOURCE, [ast.ClassDef(name="Script", bases=[], keywords=[], body=methods, decorator_list=[])], namespace)
    return namespace["Script"], enums, written


def test_controlnet_extension_reads_indexed_legacy_remote_units():
    from types import SimpleNamespace

    p = SimpleNamespace(
        control_net_enabled=True, control_net_module="canny", control_net_image="first-image",
        control_net_pres=[256, 384],  # a list holds one value per unit
        control_net_enabled2=True, control_net_module2="depth", control_net_image2="second-image",
        control_net_weight2=0.5, control_net_resize_mode2=2, control_net_control_mode2="1",
        control_net_guidance_start2=0.7, control_net_guidance_end2=0.4,
        control_net_guidance_end=1.5,
    )
    script, enums, written = _controlnet_remote_reader(allow_script_control=True)
    units = script.get_enabled_units(p)

    # A scalar control_net_enabled enables unit 1 only; unit 3 has no enabled field of its own.
    assert [(unit.module, unit.image) for unit in units] == [("canny", "first-image"), ("depth", "second-image")]
    assert [unit.weight for unit in units] == [1.0, 0.5]
    assert [unit.processor_res for unit in units] == [256, 384]
    assert [unit.resize_mode for unit in units] == [enums.ResizeMode.INNER_FIT, enums.ResizeMode.OUTER_FIT]
    assert [unit.control_mode for unit in units] == [enums.ControlMode.BALANCED, enums.ControlMode.PROMPT]
    # Other scalar fields apply to every unit without an indexed value; the interval is clamped and ordered.
    assert [(unit.guidance_start, unit.guidance_end) for unit in units] == [(0.0, 1.0), (0.7, 0.7)]
    assert written == [units]

    script, _enums, _written = _controlnet_remote_reader(allow_script_control=False)
    assert script.get_enabled_units(p) == []
