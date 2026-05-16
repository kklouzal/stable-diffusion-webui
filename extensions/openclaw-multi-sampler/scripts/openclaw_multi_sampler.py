from __future__ import annotations

import inspect
import json
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from modules import headless_ui as gr
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, Request

import k_diffusion.sampling
from modules import devices, script_callbacks, scripts, sd_samplers, sd_samplers_common, sd_samplers_kdiffusion, shared
from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback
from modules.shared import opts, state
try:
    from openclaw_denoise_ramp import ramp_sigmas_for_img2img
except Exception:
    ramp_sigmas_for_img2img = None

EXT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXT_ROOT / "data"
CUSTOM_FILE = DATA_DIR / "custom_samplers.json"
SNAPSHOT_ROOT = DATA_DIR / "snapshots"
CUSTOM_PREFIX = "Multi: "
PREVIEW_NAME = f"{CUSTOM_PREFIX}Controller Preview"
_LOCK = threading.RLock()
_REGISTERED_NAMES: set[str] = set()
_TRANSIENT_DEFS: dict[str, dict[str, Any]] = {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text or "").strip()).strip("-._")
    return slug[:80] or "multi-sampler"


def _k_sampler_names() -> list[str]:
    return [item[0] for item in sd_samplers_kdiffusion.samplers_k_diffusion]


def _k_sampler_config(name: str):
    config = sd_samplers_kdiffusion.k_diffusion_samplers_map.get(name)
    if config is None:
        raise ValueError(f"Unsupported k-diffusion sampler: {name}")
    return config


def _chain_sampler_names(definition: dict[str, Any]) -> list[str]:
    raw = definition.get("samplers")
    if isinstance(raw, (list, tuple)):
        names = [str(item).strip() for item in raw if str(item).strip()]
    else:
        names = []
        for index in range(1, 4):
            value = definition.get(f"sampler_{index}")
            if value is None:
                value = definition.get(f"sampler{index}")
            if value is None:
                continue
            value = str(value).strip()
            if value:
                names.append(value)
    if len(names) < 2:
        raise ValueError("At least two sampler stages are required")
    for name in names:
        _k_sampler_config(name)
    return names


def _chain_switch_points(definition: dict[str, Any], stage_count: int) -> list[int]:
    raw = definition.get("switch_ats")
    if isinstance(raw, (list, tuple)):
        points = [_safe_int(item, 0) for item in raw]
    else:
        points = []
        for index in range(1, stage_count):
            if index == 1:
                value = definition.get("switch_at")
                if value is None:
                    value = 10
            else:
                value = definition.get(f"switch_at_{index}")
                if value is None:
                    value = definition.get(f"switch_at{index}")
            if value is None:
                raise ValueError(f"Switch point {index} is required for a {stage_count}-stage sampler chain")
            points.append(_safe_int(value, 0))
    if len(points) != stage_count - 1:
        raise ValueError(f"Expected {stage_count - 1} switch point(s) for a {stage_count}-stage sampler chain")
    prev = 0
    for point in points:
        if point < prev:
            raise ValueError("Switch points must be in nondecreasing order")
        prev = point
    return points


def _chain_boundaries(definition: dict[str, Any], steps: int) -> tuple[list[str], list[int]]:
    samplers = _chain_sampler_names(definition)
    switch_ats = _chain_switch_points(definition, len(samplers))
    boundaries = [0]
    for point in switch_ats:
        boundaries.append(max(boundaries[-1], min(_safe_int(point, 0), steps)))
    boundaries.append(steps)
    return samplers, boundaries


def _definition_stages(definition: dict[str, Any], sigmas: torch.Tensor, steps: int) -> list[tuple[str, torch.Tensor, int, int]]:
    samplers, boundaries = _chain_boundaries(definition, steps)
    return [
        (sampler_name, sigmas[boundaries[index] : boundaries[index + 1] + 1], boundaries[index], boundaries[index + 1])
        for index, sampler_name in enumerate(samplers)
    ]


def _format_chain_metadata(stages: list[tuple[str, torch.Tensor, int, int]]) -> str:
    return " -> ".join(f"{sampler_name}@{start}-{end}" for sampler_name, _stage_sigmas, start, end in stages)


def _normalize_definition(data: dict[str, Any], *, require_name: bool = True) -> dict[str, Any]:
    name = str(data.get("name") or "").strip()
    if require_name and not name:
        raise ValueError("Custom sampler name is required")
    if name and not name.startswith(CUSTOM_PREFIX):
        name = f"{CUSTOM_PREFIX}{name}"
    sampler_names = _chain_sampler_names(data)
    switch_ats = _chain_switch_points(data, len(sampler_names))
    definition = {
        "name": name,
        "samplers": sampler_names,
        "switch_ats": switch_ats,
        "created_at": float(data.get("created_at") or time.time()),
        "updated_at": time.time(),
    }
    for index, sampler_name in enumerate(sampler_names, start=1):
        definition[f"sampler_{index}"] = sampler_name
    for index, switch_at in enumerate(switch_ats, start=1):
        key = "switch_at" if index == 1 else f"switch_at_{index}"
        definition[key] = switch_at
    return definition


def _load_custom_defs() -> list[dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CUSTOM_FILE.exists():
        return []
    try:
        raw = json.loads(CUSTOM_FILE.read_text(encoding="utf-8"))
        items = raw.get("samplers") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return []
        normalized = []
        for item in items:
            if isinstance(item, dict):
                try:
                    normalized.append(_normalize_definition(item))
                except Exception as exc:
                    print(f"[openclaw-multi-sampler] ignoring invalid sampler definition: {exc}")
        return normalized
    except Exception as exc:
        print(f"[openclaw-multi-sampler] failed to read {CUSTOM_FILE}: {exc}")
        return []


def _save_custom_defs(defs: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_FILE.write_text(json.dumps({"samplers": defs}, indent=2, sort_keys=True), encoding="utf-8")


def _stage_extra_params(funcname: str) -> list[str]:
    return sd_samplers_kdiffusion.sampler_extra_params.get(funcname, [])


class MultiKDiffusionSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    """Run a multi-stage k-diffusion sampler chain over one continuous sigma schedule."""

    def __init__(self, sd_model, definition: dict[str, Any]):
        # Use Euler as a harmless base function; sample/sample_img2img are overridden.
        super().__init__("sample_euler", sd_model)
        self.definition = dict(definition)
        self.extra_params = []

    def _split_sigmas(self, sigmas: torch.Tensor, steps: int) -> list[tuple[str, torch.Tensor, int, int]]:
        return _definition_stages(self.definition, sigmas, steps)

    def _initialize_chain(self, p) -> None:
        self.p = p
        self.model_wrap_cfg.p = p
        self.model_wrap_cfg.mask = p.mask if hasattr(p, "mask") else None
        self.model_wrap_cfg.nmask = p.nmask if hasattr(p, "nmask") else None
        self.model_wrap_cfg.step = 0
        self.model_wrap_cfg.image_cfg_scale = getattr(p, "image_cfg_scale", None)
        self.eta = p.eta if p.eta is not None else getattr(opts, self.eta_option_field, 0.0)
        self.s_min_uncond = getattr(p, "s_min_uncond", 0.0)
        k_diffusion.sampling.torch = sd_samplers_common.TorchHijack(p)

    def _build_stage_kwargs(self, *, p, func, funcname: str, config, x, sigmas: torch.Tensor, full_sigmas: torch.Tensor, stage_steps: int, is_img2img: bool) -> dict[str, Any]:
        params = inspect.signature(func).parameters
        kwargs: dict[str, Any] = {}
        for param_name in _stage_extra_params(funcname):
            if param_name not in params:
                continue
            value = getattr(p, param_name, None)
            if param_name == "s_churn":
                value = getattr(opts, "s_churn", getattr(p, "s_churn", 0.0))
            elif param_name == "s_tmin":
                value = getattr(opts, "s_tmin", getattr(p, "s_tmin", 0.0))
            elif param_name == "s_tmax":
                value = getattr(opts, "s_tmax", getattr(p, "s_tmax", float("inf"))) or float("inf")
            elif param_name == "s_noise":
                value = getattr(opts, "s_noise", getattr(p, "s_noise", 1.0))
            kwargs[param_name] = value
        if "eta" in params:
            kwargs["eta"] = self.eta
        if "n" in params:
            kwargs["n"] = stage_steps
        if "sigma_min" in params:
            positive_sigmas = sigmas[sigmas > 0]
            kwargs["sigma_min"] = positive_sigmas[-1] if len(positive_sigmas) else sigmas[-1]
            kwargs["sigma_max"] = sigmas[0]
        if "sigma_sched" in params:
            kwargs["sigma_sched"] = sigmas
        if "sigmas" in params:
            kwargs["sigmas"] = sigmas
        if config.options.get("brownian_noise", False):
            kwargs["noise_sampler"] = self.create_noise_sampler(x, full_sigmas, p)
        if config.options.get("solver_type", None) == "heun":
            kwargs["solver_type"] = "heun"
        return kwargs

    def _snapshot_config(self, p) -> dict[str, Any]:
        raw = getattr(p, "openclaw_multi_sampler_snapshots", None)
        return raw if isinstance(raw, dict) and raw.get("enabled") else {}

    def _save_snapshot(self, p, latent: torch.Tensor, *, step: int, final: bool = False) -> None:
        cfg = self._snapshot_config(p)
        if not cfg:
            return
        every = max(1, _safe_int(cfg.get("every"), 1))
        max_count = max(0, _safe_int(cfg.get("max_count"), 0))
        if not final and step % every != 0:
            return
        if max_count and not final and step // every > max_count:
            return
        out_dir = Path(str(cfg.get("dir") or "")).expanduser()
        if not out_dir:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        approximation = cfg.get("approximation")
        if approximation in ("Full", "full", 0, "0"):
            approx = 0
        elif approximation in ("Approx NN", "approx", 1, "1"):
            approx = 1
        elif approximation in ("Approx cheap", "cheap", 2, "2", None, ""):
            approx = 2
        elif approximation in ("TAESD", "taesd", 3, "3"):
            approx = 3
        else:
            approx = 2
        try:
            tensor = sd_samplers_common.samples_to_images_tensor(latent.detach().float(), approximation=approx)[0] * 0.5 + 0.5
            tensor = torch.clamp(tensor, min=0.0, max=1.0).float()
            array = (255.0 * np.moveaxis(tensor.cpu().numpy(), 0, 2)).astype(np.uint8)
            image = Image.fromarray(array)
            name = "final.png" if final else f"step-{step:03d}.png"
            image.save(out_dir / name)
        except Exception:
            print("[openclaw-multi-sampler] snapshot save failed")
            traceback.print_exc()

    def _callback(self, p, *, offset: int):
        def inner(d: dict[str, Any]):
            local_step = _safe_int(d.get("i"), 0)
            global_step = offset + local_step
            if self.stop_at is not None and global_step > self.stop_at:
                raise sd_samplers_common.InterruptedException
            state.sampling_step = global_step
            shared.total_tqdm.update()
            denoised = d.get("denoised")
            if denoised is not None:
                self._save_snapshot(p, denoised, step=global_step + 1)
        return inner

    def _stage_total_steps(self, sampler_name: str, stage_steps: int) -> int:
        return _k_sampler_config(sampler_name).total_steps(stage_steps)

    def _run_chain(self, p, x, conditioning, unconditional_conditioning, sigmas: torch.Tensor, steps: int, *, is_img2img: bool, image_conditioning=None):
        self._initialize_chain(p)
        stages = self._split_sigmas(sigmas, steps)
        self.model_wrap_cfg.steps = steps
        self.model_wrap_cfg.total_steps = sum(self._stage_total_steps(name, max(0, end - start)) for name, _stage_sigmas, start, end in stages)
        state.sampling_steps = steps
        state.sampling_step = 0
        self.last_latent = x
        self.sampler_extra_args = {
            "cond": conditioning,
            "image_cond": image_conditioning,
            "uncond": unconditional_conditioning,
            "cond_scale": p.cfg_scale,
            "s_min_uncond": self.s_min_uncond,
        }
        p.extra_generation_params["Sampler chain"] = _format_chain_metadata(stages)
        scheduler = _sampler_data_for(self.definition).options.get("scheduler")
        if scheduler:
            p.extra_generation_params["Sampler chain scheduler"] = scheduler
        try:
            for sampler_name, stage_sigmas, offset, end in stages:
                stage_steps = max(0, end - offset)
                if stage_steps <= 0:
                    continue
                config = _k_sampler_config(sampler_name)
                funcname = config.constructor.keywords.get("funcname") if hasattr(config.constructor, "keywords") else None
                # SamplerData constructors are lambdas, so map via the source table instead.
                funcname = next(item[1] for item in sd_samplers_kdiffusion.samplers_k_diffusion if item[0] == sampler_name)
                func = funcname if callable(funcname) else getattr(k_diffusion.sampling, funcname)
                stage_funcname = funcname if isinstance(funcname, str) else getattr(funcname, "__name__", "")
                kwargs = self._build_stage_kwargs(p=p, func=func, funcname=stage_funcname, config=config, x=x, sigmas=stage_sigmas, full_sigmas=sigmas, stage_steps=stage_steps, is_img2img=is_img2img)
                # k-diffusion's sample_dpmpp_2m_sde has an h_last bookkeeping bug when
                # it is asked to do only the final denoise transition [sigma, 0].
                # A mid-chain split can naturally create that one-step stage, so handle
                # it explicitly instead of rejecting useful takeover points.
                if stage_funcname == "sample_dpmpp_2m_sde" and stage_steps == 1 and float(stage_sigmas[-1]) == 0.0:
                    s_in = x.new_ones([x.shape[0]])
                    denoised = self.model_wrap_cfg(x, stage_sigmas[0] * s_in, **self.sampler_extra_args)
                    self._callback(p, offset=offset)({"x": x, "i": 0, "sigma": stage_sigmas[0], "sigma_hat": stage_sigmas[0], "denoised": denoised})
                    self.last_latent = denoised
                    x = denoised
                else:
                    x = func(self.model_wrap_cfg, x, extra_args=self.sampler_extra_args, disable=False, callback=self._callback(p, offset=offset), **kwargs)
                self.last_latent = x
            self._save_snapshot(p, x, step=steps, final=True)
            return x
        except RecursionError:
            print("Encountered RecursionError during multi-sampler sampling; returning last latent.")
            return self.last_latent
        except sd_samplers_common.InterruptedException:
            return self.last_latent

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        steps = steps or p.steps
        sigmas = self.get_sigmas(p, steps)
        if opts.sgm_noise_multiplier:
            p.extra_generation_params["SGM noise multiplier"] = True
            x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        else:
            x = x * sigmas[0]
        samples = self._run_chain(p, x, conditioning, unconditional_conditioning, sigmas, steps, is_img2img=False, image_conditioning=image_conditioning)
        self.add_infotext(p)
        return samples

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)
        sigmas = self.get_sigmas(p, steps)
        if ramp_sigmas_for_img2img is not None:
            sigmas = ramp_sigmas_for_img2img(p, sigmas, steps, t_enc)
        sigma_sched = sigmas[steps - t_enc - 1:]
        if hasattr(shared.sd_model, "add_noise_to_latent"):
            xi = shared.sd_model.add_noise_to_latent(x, noise, sigma_sched[0])
        else:
            xi = x + noise * sigma_sched[0]

        if opts.img2img_extra_noise > 0:
            p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
            extra_noise_params = ExtraNoiseParams(noise, x, xi)
            extra_noise_callback(extra_noise_params)
            noise = extra_noise_params.noise
            xi += noise * opts.img2img_extra_noise

        self.model_wrap_cfg.init_latent = x
        self.last_latent = x
        samples = self._run_chain(p, xi, conditioning, unconditional_conditioning, sigma_sched, t_enc, is_img2img=True, image_conditioning=image_conditioning)
        self.add_infotext(p)
        return samples


class MultiSamplerData(sd_samplers_common.SamplerData):
    def total_steps(self, steps):
        chain = self.options.get("openclaw_chain") or {}
        samplers, boundaries = _chain_boundaries(chain, steps)
        total = 0
        for index, sampler_name in enumerate(samplers):
            total += _k_sampler_config(sampler_name).total_steps(max(0, boundaries[index + 1] - boundaries[index]))
        return total


def _sampler_data_for(definition: dict[str, Any]) -> sd_samplers_common.SamplerData:
    name = definition["name"]
    chain = dict(definition)
    configs = [_k_sampler_config(sampler_name) for sampler_name in _chain_sampler_names(chain)]
    scheduler = next((config.options.get("scheduler") for config in configs if config.options.get("scheduler")), "karras")
    opts_union = {"scheduler": scheduler, "openclaw_chain": chain}
    if any(config.options.get("uses_ensd") for config in configs):
        opts_union["uses_ensd"] = True
    if any(config.options.get("brownian_noise") for config in configs):
        opts_union["brownian_noise"] = True
    return MultiSamplerData(name, lambda model, chain=chain: MultiKDiffusionSampler(model, chain), [], opts_union)


def _register_definitions() -> None:
    with _LOCK:
        defs = _load_custom_defs() + list(_TRANSIENT_DEFS.values())
        for name in list(_REGISTERED_NAMES):
            sd_samplers.all_samplers_map.pop(name, None)
            sd_samplers.all_samplers[:] = [s for s in sd_samplers.all_samplers if s.name != name]
        _REGISTERED_NAMES.clear()
        for definition in defs:
            data = _sampler_data_for(definition)
            sd_samplers.all_samplers.append(data)
            sd_samplers.all_samplers_map[data.name] = data
            _REGISTERED_NAMES.add(data.name)
        sd_samplers.set_samplers()


def _upsert_custom(definition: dict[str, Any]) -> dict[str, Any]:
    with _LOCK:
        normalized = _normalize_definition(definition)
        defs = _load_custom_defs()
        defs = [item for item in defs if item.get("name") != normalized["name"]]
        defs.append(normalized)
        defs.sort(key=lambda item: item.get("name", ""))
        _save_custom_defs(defs)
        _register_definitions()
        return normalized


class OpenClawMultiSamplerScript(scripts.Script):
    def title(self):
        return "OpenClaw Multi-Sampler"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        enabled = gr.Checkbox(value=False, visible=False, label="Enable Multi-Sampler snapshots")
        snapshot_dir = gr.Textbox(value="", visible=False, label="Snapshot directory")
        every = gr.Number(value=1, visible=False, precision=0, label="Snapshot every N steps")
        max_count = gr.Number(value=0, visible=False, precision=0, label="Max snapshots")
        approximation = gr.Dropdown(choices=["Approx cheap", "Approx NN", "TAESD", "Full"], value="Approx cheap", visible=False, label="Snapshot decoder")
        return [enabled, snapshot_dir, every, max_count, approximation]

    def process(self, p, enabled=False, snapshot_dir="", every=1, max_count=0, approximation="Approx cheap"):
        if enabled and snapshot_dir:
            p.openclaw_multi_sampler_snapshots = {
                "enabled": True,
                "dir": str(snapshot_dir),
                "every": _safe_int(every, 1) or 1,
                "max_count": _safe_int(max_count, 0),
                "approximation": approximation or "Approx cheap",
            }
        else:
            p.openclaw_multi_sampler_snapshots = {"enabled": False}


def on_app_started(_: object, app: FastAPI) -> None:
    _register_definitions()

    @app.get("/sdapi/v1/openclaw/multi-sampler")
    async def list_multi_samplers():
        return {"ok": True, "k_samplers": _k_sampler_names(), "custom_samplers": _load_custom_defs(), "snapshot_root": str(SNAPSHOT_ROOT)}

    @app.post("/sdapi/v1/openclaw/multi-sampler/custom")
    async def save_multi_sampler(request: Request):
        try:
            data = await request.json()
            definition = _upsert_custom(data)
            return {"ok": True, "sampler": definition}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}

    @app.delete("/sdapi/v1/openclaw/multi-sampler/custom/{name:path}")
    async def delete_multi_sampler(name: str):
        full_name = name if name.startswith(CUSTOM_PREFIX) else f"{CUSTOM_PREFIX}{name}"
        defs = [item for item in _load_custom_defs() if item.get("name") != full_name]
        _save_custom_defs(defs)
        _register_definitions()
        return {"ok": True, "deleted": full_name}

    @app.post("/sdapi/v1/openclaw/multi-sampler/preview")
    async def preview_multi_sampler(request: Request):
        try:
            data = await request.json()
            definition = _normalize_definition({**data, "name": PREVIEW_NAME})
            _TRANSIENT_DEFS[PREVIEW_NAME] = definition
            _register_definitions()
            run_id = _slug(str(data.get("run_id") or f"run-{int(time.time())}"))
            snapshot_dir = SNAPSHOT_ROOT / run_id
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            return {"ok": True, "sampler": definition, "snapshot_dir": str(snapshot_dir), "sampler_name": PREVIEW_NAME}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


_register_definitions()
script_callbacks.on_app_started(on_app_started)
