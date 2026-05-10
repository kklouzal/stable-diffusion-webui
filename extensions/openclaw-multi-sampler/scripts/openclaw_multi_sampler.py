from __future__ import annotations

import inspect
import json
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, Request

import k_diffusion.sampling
from modules import devices, script_callbacks, scripts, sd_samplers, sd_samplers_common, sd_samplers_kdiffusion, shared
from modules.shared import opts, state

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


def _normalize_definition(data: dict[str, Any], *, require_name: bool = True) -> dict[str, Any]:
    name = str(data.get("name") or "").strip()
    if require_name and not name:
        raise ValueError("Custom sampler name is required")
    if name and not name.startswith(CUSTOM_PREFIX):
        name = f"{CUSTOM_PREFIX}{name}"
    sampler_1 = str(data.get("sampler_1") or data.get("sampler1") or "Euler a").strip()
    sampler_2 = str(data.get("sampler_2") or data.get("sampler2") or "DPM++ 2M SDE").strip()
    _k_sampler_config(sampler_1)
    _k_sampler_config(sampler_2)
    switch_at = max(0, _safe_int(data.get("switch_at"), 10))
    return {
        "name": name,
        "sampler_1": sampler_1,
        "sampler_2": sampler_2,
        "switch_at": switch_at,
        "created_at": float(data.get("created_at") or time.time()),
        "updated_at": time.time(),
    }


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
    """Run two k-diffusion sampler functions over one continuous sigma schedule."""

    def __init__(self, sd_model, definition: dict[str, Any]):
        # Use Euler as a harmless base function; sample/sample_img2img are overridden.
        super().__init__("sample_euler", sd_model)
        self.definition = dict(definition)
        self.extra_params = []

    def _split_sigmas(self, sigmas: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        switch_at = max(0, min(_safe_int(self.definition.get("switch_at"), 0), steps))
        # If switch_at is 10, stage 1 performs transitions 0..9 on sigmas[0:11];
        # stage 2 resumes from the shared boundary sigma at sigmas[10].
        first = sigmas[: switch_at + 1]
        second = sigmas[switch_at:]
        return first, second, switch_at

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
            if is_img2img:
                kwargs["sigma_min"] = sigmas[-2] if len(sigmas) > 1 else sigmas[-1]
                kwargs["sigma_max"] = sigmas[0]
            else:
                kwargs["sigma_min"] = self.model_wrap.sigmas[0].item()
                kwargs["sigma_max"] = self.model_wrap.sigmas[-1].item()
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

    def _run_chain(self, p, x, conditioning, unconditional_conditioning, sigmas: torch.Tensor, steps: int, *, is_img2img: bool, image_conditioning=None):
        self._initialize_chain(p)
        first_sigmas, second_sigmas, switch_at = self._split_sigmas(sigmas, steps)
        stages = [
            (self.definition["sampler_1"], first_sigmas, 0),
            (self.definition["sampler_2"], second_sigmas, switch_at),
        ]
        self.model_wrap_cfg.steps = steps
        self.model_wrap_cfg.total_steps = steps
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
        p.extra_generation_params["Sampler chain"] = "{}@{} -> {}@{}".format(self.definition["sampler_1"], switch_at, self.definition["sampler_2"], steps - switch_at)
        try:
            for sampler_name, stage_sigmas, offset in stages:
                stage_steps = max(0, len(stage_sigmas) - 1)
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
        sigma_sched = sigmas[steps - t_enc - 1:]
        if hasattr(shared.sd_model, "add_noise_to_latent"):
            xi = shared.sd_model.add_noise_to_latent(x, noise, sigma_sched[0])
        else:
            xi = x + noise * sigma_sched[0]
        self.model_wrap_cfg.init_latent = x
        self.last_latent = x
        samples = self._run_chain(p, xi, conditioning, unconditional_conditioning, sigma_sched, t_enc, is_img2img=True, image_conditioning=image_conditioning)
        self.add_infotext(p)
        return samples


def _sampler_data_for(definition: dict[str, Any]) -> sd_samplers_common.SamplerData:
    name = definition["name"]
    chain = dict(definition)
    first = _k_sampler_config(chain["sampler_1"])
    second = _k_sampler_config(chain["sampler_2"])
    opts_union = {"scheduler": first.options.get("scheduler") or second.options.get("scheduler") or "karras"}
    if first.options.get("uses_ensd") or second.options.get("uses_ensd"):
        opts_union["uses_ensd"] = True
    if first.options.get("second_order") or second.options.get("second_order"):
        opts_union["second_order"] = True
    if first.options.get("brownian_noise") or second.options.get("brownian_noise"):
        opts_union["brownian_noise"] = True
    return sd_samplers_common.SamplerData(name, lambda model, chain=chain: MultiKDiffusionSampler(model, chain), [], opts_union)


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
