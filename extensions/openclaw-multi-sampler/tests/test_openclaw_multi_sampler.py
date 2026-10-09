from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

# The registry-mutation watcher shared with the core sampler-registry test (test/helpers.py imports only the stdlib).
from test.helpers import WatchingDict

EXT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXT_ROOT.parents[1]
SCRIPT_PATH = EXT_ROOT / "scripts" / "openclaw_multi_sampler.py"


def _install_a1111_stubs(monkeypatch) -> None:
    """Minimal A1111/k-diffusion surface the script imports; monkeypatch restores sys.modules after each test."""

    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    class SamplerData:
        def __init__(self, name, constructor, aliases=None, options=None):
            self.name = name
            self.constructor = constructor
            self.aliases = aliases or []
            self.options = options or {}

    class FakeSamplerConfig:
        def __init__(self, name, options=None):
            self.name = name
            self.options = options or {}

        @staticmethod
        def total_steps(steps):
            return steps

    sampler_configs = {
        "Euler": FakeSamplerConfig("Euler"),
        "DPM++ 2M SDE": FakeSamplerConfig("DPM++ 2M SDE", {"brownian_noise": True}),
        "Heun": FakeSamplerConfig("Heun", {"solver_type": "heun"}),
        "DPM2": FakeSamplerConfig("DPM2", {"discard_next_to_last_sigma": True}),
    }

    class KDiffusionSampler:
        def __init__(self, _funcname, _sd_model):
            self.config = sampler_configs["Euler"]
            self.model_wrap_cfg = types.SimpleNamespace()
            self.eta_option_field = "eta"
            self.eta = 0.0
            self.stop_at = None

        def initialize(self, _p):
            return {}

        def set_sampler_extra_args(self, p, conditioning, unconditional_conditioning, image_conditioning):
            self.sampler_extra_args = {"p": p, "cond": conditioning, "uncond": unconditional_conditioning, "image_cond": image_conditioning}
            return self.sampler_extra_args

        def get_sigmas(self, _p, steps):
            return torch.linspace(float(steps), 0.0, steps + 1)

        def sample_img2img(self, *args, **kwargs):
            return args, kwargs

        def create_noise_sampler(self, *_args, **_kwargs):
            return None

        def add_infotext(self, _p):
            return None

    module("modules")
    module("modules.headless_ui")
    module("modules.script_callbacks", on_app_started=lambda _callback: None)
    module("modules.script_loading", loaded_scripts={})
    module("modules.scripts", Script=object, AlwaysVisible=object())
    module(
        "modules.sd_samplers_common",
        SamplerData=SamplerData,
        InterruptedException=type("InterruptedException", (Exception,), {}),
        setup_img2img_steps=lambda p, steps=None: (steps or p.steps, getattr(p, "t_enc", steps or p.steps)),
        samples_to_images_tensor=lambda latent, approximation=2: latent,
    )
    module("modules.sd_samplers", all_samplers=[], all_samplers_map={}, set_samplers=lambda: None)
    module(
        "modules.sd_schedulers",
        schedulers_map={"Automatic": object(), "Karras": object(), "Exponential": object(), "Normal": object()},
        schedulers=[types.SimpleNamespace(label=label) for label in ("Automatic", "Karras", "Exponential", "Normal")],
    )
    module(
        "modules.shared",
        opts=types.SimpleNamespace(s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, sgm_noise_multiplier=False),
        cmd_opts=types.SimpleNamespace(disable_console_progressbars=True),
        state=types.SimpleNamespace(sampling_step=0, sampling_steps=0),
        total_tqdm=types.SimpleNamespace(update=lambda: None),
    )
    module(
        "modules.sd_samplers_kdiffusion",
        KDiffusionSampler=KDiffusionSampler,
        samplers_k_diffusion=[
            ("Euler", "sample_euler"),
            ("DPM++ 2M SDE", "sample_dpmpp_2m_sde"),
            ("Heun", "sample_heun"),
            ("DPM2", "sample_dpm_2"),
        ],
        k_diffusion_samplers_map=sampler_configs,
        sampler_extra_params={
            "sample_euler": ["s_churn", "s_tmin", "s_tmax", "s_noise"],
            "sample_dpmpp_2m_sde": [],
            "sample_heun": [],
            "sample_dpm_2": [],
        },
    )
    sampling = module(
        "k_diffusion.sampling",
        sample_euler=lambda _model, x, **_kwargs: x,
        sample_dpmpp_2m_sde=lambda _model, x, **_kwargs: x,
        sample_heun=lambda _model, x, **_kwargs: x,
        sample_dpm_2=lambda _model, x, **_kwargs: x,
    )
    module("k_diffusion", sampling=sampling)


@pytest.fixture
def multi(monkeypatch):
    """A fresh module per test: registry, caches and the denoise-ramp lookup state never leak between tests."""
    _install_a1111_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location("openclaw_multi_sampler_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chain_boundaries_reject_zero_length_stages(multi):
    with pytest.raises(ValueError, match="stage 1"):
        multi._chain_boundaries({"samplers": ["Euler", "Heun"], "switch_ats": [0]}, 6)

    # The switch point is clamped to the step count, which leaves stage 2 empty.
    with pytest.raises(ValueError, match="stage 2 has no sampling steps"):
        multi._chain_boundaries({"samplers": ["Euler", "Heun"], "switch_ats": [25]}, 20)

    with pytest.raises(ValueError, match="at least one sampling step"):
        multi._chain_boundaries({"samplers": ["Euler", "Heun"], "switch_ats": [0]}, 0)


def test_normalize_definition_preserves_arbitrary_length_scheduler_chain(multi):
    definition = multi._normalize_definition(
        {
            "name": "three stage",
            "samplers": ["Euler", "Heun", "DPM++ 2M SDE"],
            "switch_ats": [4, 8],
            "schedulers": ["Karras", "Exponential", "Normal"],
            "created_at": 123.0,
        }
    )

    assert definition["name"] == "Multi: three stage"
    assert definition["samplers"] == ["Euler", "Heun", "DPM++ 2M SDE"]
    assert definition["switch_ats"] == [4, 8]
    assert definition["schedulers"] == ["Karras", "Exponential", "Normal"]
    assert definition["created_at"] == 123.0
    assert definition["sampler_3"] == "DPM++ 2M SDE"
    assert definition["scheduler_3"] == "Normal"


def test_img2img_sigma_tail_transition_count_keeps_terminal_sigma(multi):
    sigma_sched = [6, 5, 4, 3, 2, 1, 0]
    sampling_steps = multi._sigma_transition_count(sigma_sched)

    sampler = object.__new__(multi.MultiKDiffusionSampler)
    sampler.definition = {"samplers": ["Euler", "Heun"], "switch_ats": [3]}

    stages = sampler._build_stages(None, sigma_sched, sampling_steps)

    assert sampling_steps == 6
    assert stages[0] == ("Euler", None, [6, 5, 4, 3], 0, 3)
    assert stages[1] == ("Heun", None, [3, 2, 1, 0], 3, 6)


def test_stage_sigmas_must_cover_expected_span(multi):
    sampler = object.__new__(multi.MultiKDiffusionSampler)
    sampler.definition = {"samplers": ["Euler", "Heun"], "switch_ats": [2]}

    with pytest.raises(ValueError, match="expected 3 sigma value"):
        sampler._build_stages(None, [4, 3, 2, 1], 4)


def _scheduler_chain_sampler(multi, switch_ats, sources):
    sampler = object.__new__(multi.MultiKDiffusionSampler)
    sampler.definition = {"samplers": ["Euler", "Heun"], "switch_ats": switch_ats, "schedulers": list(sources)}
    sampler._sigmas_for_scheduler = lambda _p, _source_steps, _sampler_name, scheduler_name: sources[scheduler_name]
    return sampler


def test_scheduler_stage_split_preserves_continuous_boundary_sigma(multi):
    sampler = _scheduler_chain_sampler(multi, [2], {
        "Karras": torch.tensor([10.0, 9.0, 8.0, 7.0, 0.0]),
        "Exponential": torch.tensor([100.0, 90.0, 80.0, 70.0, 0.0]),
    })

    stages = sampler._build_stages(types.SimpleNamespace(), torch.zeros(5), steps=4)

    torch.testing.assert_close(stages[0][2], torch.tensor([10.0, 9.0, 8.0]))
    # The handoff sigma (8) is below the later stage's last positive sigma (70), so the stage is scaled down whole
    # instead of rising from 8 back up to 70.
    torch.testing.assert_close(stages[1][2], torch.tensor([8.0, 7.0, 0.0]))


def test_scheduler_stage_split_keeps_non_rising_splice_bit_identical(multi):
    sampler = _scheduler_chain_sampler(multi, [2], {
        "Karras": torch.tensor([10.0, 9.0, 8.0, 7.0, 0.0]),
        "Exponential": torch.tensor([12.0, 9.5, 8.5, 3.0, 0.0]),
    })

    stages = sampler._build_stages(types.SimpleNamespace(), torch.zeros(5), steps=4)

    assert torch.equal(stages[1][2], torch.tensor([8.0, 3.0, 0.0]))


def test_scheduler_stage_split_rebases_rising_handoff_in_log_sigma(multi):
    # "Multi: oi" img2img: Euler a [Exponential] ends at 0.1011, DPM++ 2M SDE [Align Your Steps] stage starts at 0.234.
    sampler = _scheduler_chain_sampler(multi, [1], {
        "Exponential": torch.tensor([0.2, 0.1011, 0.09, 0.05, 0.02, 0.01, 0.0]),
        "Karras": torch.tensor([0.5, 0.234, 0.1626, 0.113, 0.0572, 0.029, 0.0]),
    })

    stage = sampler._build_stages(types.SimpleNamespace(), torch.zeros(7), steps=6)[1][2]

    assert float(stage[0]) == float(torch.tensor(0.1011))
    assert float(stage[-2]) == pytest.approx(0.029, abs=5e-7)
    assert float(stage[-1]) == 0.0
    assert bool((stage[1:] < stage[:-1]).all())
    torch.testing.assert_close(stage, torch.tensor([0.1011, 0.0812, 0.0654, 0.0435, 0.029, 0.0]), atol=2e-4, rtol=0)


def test_stage_sigma_validation_rejects_rising_or_non_finite_sigmas(multi):
    with pytest.raises(ValueError, match="rise"):
        multi._validate_stage_sigmas(torch.tensor([0.0, 0.5, 0.0]), 0, 2, "Euler", None)
    with pytest.raises(ValueError, match="non-finite"):
        multi._validate_stage_sigmas(torch.tensor([1.0, float("nan"), 0.0]), 0, 2, "Euler", None)


def test_brownian_noise_sampler_uses_stage_sigmas_not_full_chain(multi):
    sampler = object.__new__(multi.MultiKDiffusionSampler)
    stage_sigmas = torch.tensor([3.0, 2.0, 0.0])
    seen = []
    sampler.create_noise_sampler = lambda _x, sigmas, _p: seen.append(sigmas) or "noise"

    kwargs = sampler._build_stage_kwargs(
        p=types.SimpleNamespace(),
        func=lambda *args, **inner_kwargs: None,
        funcname="sample_dpmpp_2m_sde",
        config=types.SimpleNamespace(options={"brownian_noise": True}),
        x=torch.zeros(1),
        sigmas=stage_sigmas,
        stage_steps=2,
    )

    assert kwargs["noise_sampler"] == "noise"
    assert seen[0] is stage_sigmas


def test_multi_sampler_data_propagates_penultimate_sigma_discard(multi):
    data = multi._sampler_data_for({"name": "Multi: dpm2", "samplers": ["Euler", "DPM2"], "switch_ats": [1]})

    assert data.options["discard_next_to_last_sigma"] is True


def test_terminal_one_step_dpmpp_2m_sde_stage_runs_the_sampler_function(multi, monkeypatch):
    # The sampler function owns the [sigma, 0] fix (modules/sd_samplers_extra.py), so the chain must not special-case it.
    calls = []

    def sample_euler(model, x, extra_args=None, disable=False, callback=None):
        calls.append(("Euler", None, extra_args))
        callback({"x": x, "i": 0, "sigma": 1, "sigma_hat": 1})
        return x

    def sample_dpmpp_2m_sde(model, x, extra_args=None, disable=False, callback=None, sigmas=None, **kwargs):
        calls.append(("DPM++ 2M SDE", list(sigmas), extra_args))
        return "denoised"

    monkeypatch.setattr(multi.k_diffusion.sampling, "sample_euler", sample_euler)
    monkeypatch.setattr(multi.k_diffusion.sampling, "sample_dpmpp_2m_sde", sample_dpmpp_2m_sde)

    class FakeModelWrapCfg:
        def __call__(self, x, sigma, **kwargs):
            return "denoised"

    p = types.SimpleNamespace(cfg_scale=7.0, extra_generation_params={})
    sampler = object.__new__(multi.MultiKDiffusionSampler)
    sampler.definition = {"name": "Multi: test", "samplers": ["Euler", "DPM++ 2M SDE"], "switch_ats": [1]}
    sampler.last_latent = None
    sampler.model_wrap_cfg = FakeModelWrapCfg()
    sampler.stop_at = None

    result = sampler._run_chain(p, "latent", "cond", "uncond", sigmas=[2, 1, 0], steps=2, image_conditioning="image_cond")

    assert result == "denoised"
    assert sampler.last_latent == "denoised"
    # Every stage gets the extra args the base sampler builds for this request.
    extra_args = {"p": p, "cond": "cond", "uncond": "uncond", "image_cond": "image_cond"}
    assert calls == [("Euler", None, extra_args), ("DPM++ 2M SDE", [1, 0], extra_args)]
    assert all(call[2] is sampler.sampler_extra_args for call in calls)
    assert p.extra_generation_params["Sampler chain"] == "Euler@0-1 -> DPM++ 2M SDE@1-2"


def _float_images_to_uint8():
    """modules.sd_samplers_common.float_images_to_uint8, compiled alone (the harness stubs that module)."""
    path = REPO_ROOT / "modules" / "sd_samplers_common.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "float_images_to_uint8"]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["float_images_to_uint8"]


def test_snapshots_round_to_uint8_like_generated_images(multi, monkeypatch, tmp_path):
    # -1 + 63.6 / 127.5 is code value 63.6: truncation gave 63, rounding gives 64; the ends map to 0 and 255.
    decoded = torch.tensor([-1.0, -1.0 + 63.6 / 127.5, 1.0]).view(1, 3, 1, 1).expand(1, 3, 2, 2).contiguous()
    monkeypatch.setattr(multi.sd_samplers_common, "samples_to_images_tensor", lambda latent, approximation=None: decoded)
    monkeypatch.setattr(multi.sd_samplers_common, "float_images_to_uint8", _float_images_to_uint8(), raising=False)
    saved = []
    image = types.SimpleNamespace(save=lambda path: saved.append(path))
    monkeypatch.setattr(multi, "Image", types.SimpleNamespace(fromarray=lambda array: saved.append(array) or image))
    sampler = object.__new__(multi.MultiKDiffusionSampler)
    p = types.SimpleNamespace(openclaw_multi_sampler_snapshots={"enabled": True, "dir": str(tmp_path)})

    sampler._save_snapshot(p, torch.zeros(1, 4, 1, 1), step=0, final=True)

    array, path = saved
    assert path == tmp_path / "final.png"
    assert array.shape == (2, 2, 3) and array.dtype.name == "uint8"
    assert array[0, 0].tolist() == [0, 64, 255]


def test_reregistering_never_drops_a_chain_that_stays_registered(multi, monkeypatch):
    # Saving a custom chain re-registers every chain from an API call that does not wait for queue_lock, while
    # generations resolve sampler names: a chain that stays registered must never be absent, not even briefly.
    keep = {"name": "Multi: keep", "samplers": ["Euler", "DPM2"], "switch_ats": [1]}
    gone = {"name": "Multi: gone", "samplers": ["Euler", "DPM2"], "switch_ats": [1]}
    monkeypatch.setattr(multi, "_load_custom_defs", lambda: [keep, gone])
    multi._register_definitions()
    registry = multi.sd_samplers
    watched = WatchingDict(registry.all_samplers_map, watch="Multi: keep")
    monkeypatch.setattr(registry, "all_samplers_map", watched)
    before = watched["Multi: keep"]

    monkeypatch.setattr(multi, "_load_custom_defs", lambda: [{**keep, "switch_ats": [2]}])
    multi._register_definitions()

    assert watched.missing_after == []
    assert watched["Multi: keep"] is not before and "Multi: gone" not in watched
    assert [s.name for s in registry.all_samplers].count("Multi: keep") == 1
    assert "Multi: gone" not in [s.name for s in registry.all_samplers]


def test_denoise_ramp_helper_is_not_imported_as_fallback(multi):
    original_sample_img2img = multi.sd_samplers_kdiffusion.KDiffusionSampler.sample_img2img

    assert multi._load_denoise_ramp_func() is None

    assert multi.sd_samplers_kdiffusion.KDiffusionSampler.sample_img2img is original_sample_img2img
    assert multi._DENOISE_RAMP_FUNC_LOADED


def test_denoise_ramp_helper_uses_loaded_script_module(multi, monkeypatch):
    target = EXT_ROOT.parent / "openclaw-denoise-ramp" / "scripts" / "openclaw_denoise_ramp.py"

    def ramp(*_args, **_kwargs):
        return None

    monkeypatch.setattr(multi.script_loading, "loaded_scripts", {str(target): types.SimpleNamespace(ramp_sigmas_for_img2img=ramp)})

    assert multi._load_denoise_ramp_func() is ramp


def test_custom_sampler_mutating_routes_hold_registration_lock():
    source = SCRIPT_PATH.read_text()
    preview_marker = '    @app.post("/sdapi/v1/openclaw/multi-sampler/preview")'
    delete_block = source[source.index("    @app.delete"):source.index(preview_marker)]
    preview_block = source[source.index(preview_marker):source.index("\n\n_register_definitions()")]

    assert "with _LOCK:" in delete_block
    assert "_save_custom_defs(defs)" in delete_block
    assert delete_block.index("with _LOCK:") < delete_block.index("_save_custom_defs(defs)")
    assert "with _LOCK:" in preview_block
    assert "_TRANSIENT_DEFS[PREVIEW_NAME] = definition" in preview_block
    assert preview_block.index("with _LOCK:") < preview_block.index("_TRANSIENT_DEFS[PREVIEW_NAME] = definition")


def _recording_core_get_sigmas(multi, monkeypatch):
    """The core get_sigmas contract the chain relies on: it reads p.hr_scheduler in the hires pass and p.scheduler
    otherwise, and records the schedule label plus option-derived schedule keys."""
    used = []

    def get_sigmas(self, p, steps):
        scheduler = p.hr_scheduler if p.is_hr_pass else p.scheduler
        used.append(scheduler)
        p.extra_generation_params["Hires schedule type" if p.is_hr_pass else "Schedule type"] = scheduler
        p.extra_generation_params["Schedule rho"] = 5.0
        return torch.linspace(float(steps), 0.0, steps + 1)

    monkeypatch.setattr(multi.sd_samplers_kdiffusion.KDiffusionSampler, "get_sigmas", get_sigmas)
    return used


@pytest.mark.parametrize("is_hr_pass", [False, True])
def test_stage_scheduler_applies_in_both_passes_and_keeps_the_request_schedulers(multi, monkeypatch, is_hr_pass):
    used = _recording_core_get_sigmas(multi, monkeypatch)
    sampler = object.__new__(multi.MultiKDiffusionSampler)
    sampler.config = "base config"
    p = types.SimpleNamespace(is_hr_pass=is_hr_pass, scheduler="Karras", hr_scheduler="Normal", extra_generation_params={})

    sampler._sigmas_for_scheduler(p, 4, "Euler", "Exponential")

    assert used == ["Exponential"]
    assert (p.scheduler, p.hr_scheduler, sampler.config) == ("Karras", "Normal", "base config")


def test_stage_schedule_calls_keep_schedule_keys_but_not_the_stage_label(multi, monkeypatch):
    _recording_core_get_sigmas(multi, monkeypatch)
    sampler = object.__new__(multi.MultiKDiffusionSampler)
    sampler.config = None
    first = types.SimpleNamespace(is_hr_pass=False, scheduler="Automatic", hr_scheduler=None, extra_generation_params={"Seed": 1})
    hires = types.SimpleNamespace(is_hr_pass=True, scheduler="Automatic", hr_scheduler="Automatic",
                                  extra_generation_params={"Schedule type": "Karras", "Hires schedule type": None, "Seed": 1})

    sampler._sigmas_for_scheduler(first, 4, "Euler", "Exponential")
    sampler._sigmas_for_scheduler(hires, 4, "Euler", "Exponential")

    # The option-derived schedule key survives; the stage label neither replaces nor adds the request's.
    assert first.extra_generation_params == {"Seed": 1, "Schedule rho": 5.0}
    assert hires.extra_generation_params == {"Schedule type": "Karras", "Hires schedule type": None, "Seed": 1, "Schedule rho": 5.0}
    assert list(hires.extra_generation_params) == ["Schedule type", "Hires schedule type", "Seed", "Schedule rho"]
