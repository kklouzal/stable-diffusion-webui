import collections
import importlib
import os
import sys
import threading
import enum

import torch
import re
import safetensors.torch
from omegaconf import OmegaConf, ListConfig
from urllib import request
import ldm.modules.midas as midas

from modules import paths, shared, modelloader, devices, script_callbacks, sd_vae, sd_disable_initialization, errors, hashes, sd_models_config, sd_unet, sd_models_xl, cache, extra_networks, processing, lowvram, sd_hijack, patches, mxfp8_model_cache, mxfp8_config, nvfp4_model_cache, nvfp4_config
from modules.hashes import partial_hash_from_cache as model_hash  # noqa: F401 for backwards compatibility
from modules.timer import Timer
from modules.shared import opts
import tomesd
import numpy as np

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))

checkpoints_list = {}
checkpoint_aliases = {}
checkpoint_alisases = checkpoint_aliases  # for compatibility with old name
checkpoints_loaded = collections.OrderedDict()


class ModelType(enum.Enum):
    SD1 = 1
    SD2 = 2
    SDXL = 3
    SSD = 4
    SD3 = 5


def replace_key(d, key, new_key, value):
    keys = list(d.keys())

    d[new_key] = value

    if key not in keys:
        return d

    index = keys.index(key)
    keys[index] = new_key

    new_d = {k: d[k] for k in keys}

    d.clear()
    d.update(new_d)
    return d


class CheckpointInfo:
    def __init__(self, filename):
        self.filename = filename
        abspath = os.path.abspath(filename)
        abs_ckpt_dir = os.path.abspath(shared.cmd_opts.ckpt_dir) if shared.cmd_opts.ckpt_dir is not None else None

        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        if abs_ckpt_dir and abspath.startswith(abs_ckpt_dir):
            name = abspath.replace(abs_ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        def read_metadata():
            metadata = read_metadata_from_safetensors(filename)
            self.modelspec_thumbnail = metadata.pop('modelspec.thumbnail', None)

            return metadata

        self.metadata = {}
        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file('safetensors-metadata', "checkpoint/" + name, filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading metadata for {filename}")

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = hashes.partial_hash_from_cache(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{name}")
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'
        self.short_title = self.name_for_extra if self.shorthash is None else f'{self.name_for_extra} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, name, self.name_for_extra, f'{name} [{self.hash}]']
        if self.shorthash:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]', f'{self.name_for_extra} [{self.shorthash}]']

    def register(self):
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_aliases[id] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return

        shorthash = self.sha256[0:10]
        if self.shorthash == self.sha256[0:10]:
            return self.shorthash

        self.shorthash = shorthash

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]', f'{self.name_for_extra} [{self.shorthash}]']

        old_title = self.title
        self.title = f'{self.name} [{self.shorthash}]'
        self.short_title = f'{self.name_for_extra} [{self.shorthash}]'

        replace_key(checkpoints_list, old_title, self.title, self)
        self.register()

        return self.shorthash


try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging, CLIPModel  # noqa: F401

    logging.set_verbosity_error()
except Exception:
    pass


def setup_model():
    """called once at startup to do various one-time tasks related to SD models"""

    os.makedirs(model_path, exist_ok=True)

    enable_midas_autodownload()
    patch_given_betas()


def checkpoint_tiles(use_short=False):
    return [x.short_title if use_short else x.title for x in checkpoints_list.values()]


def list_models():
    checkpoints_list.clear()
    checkpoint_aliases.clear()

    cmd_ckpt = shared.cmd_opts.ckpt
    if shared.cmd_opts.no_download_sd_model or cmd_ckpt != shared.sd_model_file or os.path.exists(cmd_ckpt):
        model_url = None
        expected_sha256 = None
    else:
        model_url = f"{shared.hf_endpoint}/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
        expected_sha256 = '6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa'

    model_list = modelloader.load_models(model_path=model_path, model_url=model_url, command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name="v1-5-pruned-emaonly.safetensors", ext_blacklist=[".vae.ckpt", ".vae.safetensors"], hash_prefix=expected_sha256)

    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()

        shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}", file=sys.stderr)

    for filename in model_list:
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


re_strip_checksum = re.compile(r"\s*\[[^]]+]\s*$")


def get_closet_checkpoint_match(search_string):
    if not search_string:
        return None

    checkpoint_info = checkpoint_aliases.get(search_string, None)
    if checkpoint_info is not None:
        return checkpoint_info

    found = sorted([info for info in checkpoints_list.values() if search_string in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]

    search_string_without_checksum = re.sub(re_strip_checksum, '', search_string)
    found = sorted([info for info in checkpoints_list.values() if search_string_without_checksum in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]

    return None


def select_checkpoint():
    """Raises `FileNotFoundError` if no checkpoints are found."""
    model_checkpoint = shared.opts.sd_model_checkpoint

    checkpoint_info = checkpoint_aliases.get(model_checkpoint, None)
    if checkpoint_info is not None:
        return checkpoint_info

    if len(checkpoints_list) == 0:
        error_message = "No checkpoints found. When searching for checkpoints, looked at:"
        if shared.cmd_opts.ckpt is not None:
            error_message += f"\n - file {os.path.abspath(shared.cmd_opts.ckpt)}"
        error_message += f"\n - directory {model_path}"
        if shared.cmd_opts.ckpt_dir is not None:
            error_message += f"\n - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}"
        error_message += "Can't run without a checkpoint. Find and place a .ckpt or .safetensors file into any of those locations."
        raise FileNotFoundError(error_message)

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


checkpoint_dict_replacements_sd1 = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

checkpoint_dict_replacements_sd2_turbo = { # Converts SD 2.1 Turbo from SGM to LDM format.
    'conditioner.embedders.0.': 'cond_stage_model.',
}


def transform_checkpoint_dict_key(k, replacements):
    for text, replacement in replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    is_sd2_turbo = 'conditioner.embedders.0.model.ln_final.weight' in pl_sd and pl_sd['conditioner.embedders.0.model.ln_final.weight'].size()[0] == 1024

    sd = {}
    for k, v in pl_sd.items():
        if is_sd2_turbo:
            new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd2_turbo)
        else:
            new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd1)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"

        res = {}

        try:
            json_data = json_start + file.read(metadata_len-2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                res[k] = v
                if isinstance(v, str) and v[0:1] == '{':
                    try:
                        res[k] = json.loads(v)
                    except Exception:
                        pass
        except Exception:
             errors.report(f"Error reading metadata from file: {filename}", exc_info=True)

        return res


def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        device = map_location or shared.weight_load_location or devices.get_optimal_device_name()

        if not shared.opts.disable_mmap_load_safetensors:
            pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
        else:
            pl_sd = safetensors.torch.load(open(checkpoint_file, 'rb').read())
            pl_sd = {k: v.to(device) for k, v in pl_sd.items()}
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd


def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer):
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    if checkpoint_info in checkpoints_loaded:
        # use checkpoint cache
        print(f"Loading weights [{sd_model_hash}] from cache")
        # move to end as latest
        checkpoints_loaded.move_to_end(checkpoint_info)
        return checkpoints_loaded[checkpoint_info]

    print(f"Loading weights [{sd_model_hash}] from {checkpoint_info.filename}")
    res = read_state_dict(checkpoint_info.filename)
    timer.record("load weights from disk")

    return res


class SkipWritingToConfig:
    """This context manager prevents load_model_weights from writing checkpoint name to the config when it loads weight."""

    skip = False
    previous = None

    def __enter__(self):
        self.previous = SkipWritingToConfig.skip
        SkipWritingToConfig.skip = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        SkipWritingToConfig.skip = self.previous


def check_fp8(model):
    if model is None:
        return None
    if devices.get_optimal_device_name() == "mps":
        enable_fp8 = False
    elif shared.opts.fp8_storage == "Enable":
        enable_fp8 = True
    elif getattr(model, "is_sdxl", False) and shared.opts.fp8_storage == "Enable for SDXL":
        enable_fp8 = True
    else:
        enable_fp8 = False
    return enable_fp8


def check_mxfp8(model):
    if model is None:
        return None
    if devices.get_optimal_device_name() == "mps":
        enable_mxfp8 = False
    elif shared.opts.mxfp8_storage == "Enable":
        enable_mxfp8 = True
    elif getattr(model, "is_sdxl", False) and shared.opts.mxfp8_storage == "Enable for SDXL":
        enable_mxfp8 = True
    else:
        enable_mxfp8 = False
    return enable_mxfp8

def check_nvfp4(model):
    if model is None:
        return None
    if devices.get_optimal_device_name() == "mps":
        enable_nvfp4 = False
    elif shared.opts.nvfp4_storage == "Enable":
        enable_nvfp4 = True
    elif getattr(model, "is_sdxl", False) and shared.opts.nvfp4_storage == "Enable for SDXL":
        enable_nvfp4 = True
    else:
        enable_nvfp4 = False
    return enable_nvfp4


class DisableFastModelLoadingForTorchAOQuant:
    def __enter__(self):
        self.previous = None
        if getattr(shared.opts, "mxfp8_storage", "Disable") != "Disable" or getattr(shared.opts, "nvfp4_storage", "Disable") != "Disable":
            self.previous = shared.cmd_opts.disable_model_loading_ram_optimization
            shared.cmd_opts.disable_model_loading_ram_optimization = True

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.previous is not None:
            shared.cmd_opts.disable_model_loading_ram_optimization = self.previous


def check_weight_quantization_mutual_exclusion(model):
    enabled = [name for name, active in (("FP8 weight", check_fp8(model)), ("MXFP8 weight", check_mxfp8(model)), ("NVFP4 weight", check_nvfp4(model))) if active]
    if len(enabled) > 1:
        raise RuntimeError(f"Weight quantization modes are mutually exclusive; disable all but one: {', '.join(enabled)}")


def mxfp8_selected_linear_coverage():
    selected = getattr(shared.opts, "mxfp8_linear_coverage", None)
    if selected is None:
        selected = mxfp8_config.LINEAR_COVERAGE_DEFAULT
    if isinstance(selected, str):
        selected = [selected]

    valid = set(mxfp8_config.LINEAR_COVERAGE_CHOICES)
    return {item for item in selected if item in valid}


def mxfp8_linear_region(fqn):
    if fqn.startswith("first_stage_model."):
        return "vae"
    if fqn.startswith("conditioner.") or fqn.startswith("cond_stage_model."):
        return mxfp8_config.LINEAR_COVERAGE_CONDITIONER
    if ".attn1." in fqn:
        return mxfp8_config.LINEAR_COVERAGE_SELF_ATTENTION
    if ".attn2." in fqn:
        return mxfp8_config.LINEAR_COVERAGE_CROSS_ATTENTION
    if fqn.startswith("model.diffusion_model."):
        return mxfp8_config.LINEAR_COVERAGE_UNET_OTHER
    return "other"


def mxfp8_linear_policy_skip_reason(module, fqn):
    region = mxfp8_linear_region(fqn)
    if region in ("vae", "other"):
        return region
    if region not in mxfp8_selected_linear_coverage():
        return region
    return None


def mxfp8_linear_skip_reason(module, fqn):
    if not isinstance(module, torch.nn.Linear):
        return "not_linear"
    policy_reason = mxfp8_linear_policy_skip_reason(module, fqn)
    if policy_reason is not None:
        return policy_reason

    # The A1111 LoRA hook for torch.nn.MultiheadAttention mutates the parent
    # module in_proj_weight/out_proj.weight directly instead of flowing through
    # Linear.forward. When out_proj.weight is an MXTensor, the backup path
    # attempts weight.to(cpu, copy=True), which TorchAO MXTensor rejects. Leave
    # those out_proj linears BF16 so active LoRAs remain safe; ordinary Linear
    # layers still use the MXFP8 merge-then-quantize path.
    if fqn.endswith((".attn.out_proj", ".self_attn.out_proj")):
        return "multihead_attention_out_proj_lora_backup"

    return mxfp8_config.technical_linear_skip_reason(module)


def mxfp8_linear_filter(module, fqn):
    return mxfp8_linear_skip_reason(module, fqn) is None


def apply_mxfp8_weight_quantization(model, timer, source_path=None):
    if devices.dtype != torch.bfloat16:
        raise RuntimeError("MXFP8 weight requires --dtype bfloat16; TorchAO MXFP8 kernels do not support float16 activations")
    if not torch.cuda.is_available():
        raise RuntimeError("MXFP8 weight requires CUDA")

    first_stage = model.first_stage_model
    model.first_stage_model = None
    eligible = 0
    technical_compatible_linear = 0
    policy_skipped_linear = 0
    incompatible_linear = 0
    skipped_linear = 0
    skipped_reasons = {}
    policy_skipped_reasons = {}
    incompatible_reasons = {}
    skipped_names = []
    managed_modules = []
    managed_ids = set()
    for fqn, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            technical_reason = mxfp8_config.technical_linear_skip_reason(module)
            policy_reason = mxfp8_linear_policy_skip_reason(module, fqn)
            if technical_reason is None:
                technical_compatible_linear += 1
            if policy_reason is not None:
                policy_skipped_linear += 1
                policy_skipped_reasons[policy_reason] = policy_skipped_reasons.get(policy_reason, 0) + 1
                reason = policy_reason
            elif technical_reason is not None:
                incompatible_linear += 1
                incompatible_reasons[technical_reason] = incompatible_reasons.get(technical_reason, 0) + 1
                reason = technical_reason
            else:
                reason = mxfp8_linear_skip_reason(module, fqn)

            if reason is None:
                eligible += 1
                managed_modules.append((fqn, module))
                managed_ids.add(id(module))
            else:
                skipped_linear += 1
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                skipped_names.append({"name": fqn, "reason": reason, "shape": tuple(module.weight.shape) if module.weight is not None else None})
    mxfp8_filter = lambda module, fqn: id(module) in managed_ids
    cache_loaded = False
    try:
        try:
            delattr(model, "network_mxfp8_managed_modules")
        except Exception:
            pass
        for _fqn, module in managed_modules:
            module.network_mxfp8_base_weight = module.weight.detach().to(devices.cpu, copy=True)
            if module.bias is not None:
                module.network_mxfp8_base_bias = module.bias.detach().to(devices.cpu, copy=True)
            else:
                module.network_mxfp8_base_bias = None

        for attr in ("network_mxfp8_active_config_signature", "network_mxfp8_prepare_stats", "network_mxfp8_prepare_error", "network_mxfp8_active_config_ready"):
            try:
                delattr(model, attr)
            except Exception:
                pass

        selected_coverage = sorted(mxfp8_selected_linear_coverage())
        cache_loaded = mxfp8_model_cache.load_into_model(model, source_path, mxfp8_filter, shared.device, selected_coverage)
        if not cache_loaded:
            from torchao.quantization import quantize_
            config = mxfp8_config.get_mxfp8_config()
            mxfp8_config.validate_kernel_preference(config)
            quantize_(model, config, filter_fn=mxfp8_filter, device=devices.device)
            mxfp8_model_cache.save_from_model(model, source_path, mxfp8_filter, eligible, skipped_linear, skipped_reasons, selected_coverage)
        model.mxfp8_quantization_stats = {"eligible_linear": eligible, "technical_compatible_linear": technical_compatible_linear, "policy_allowed_linear": eligible, "selected_linear_coverage": selected_coverage, "policy_skipped_linear": policy_skipped_linear, "incompatible_linear": incompatible_linear, "skipped_linear": skipped_linear, "skipped_reasons": skipped_reasons, "policy_skipped_reasons": policy_skipped_reasons, "incompatible_reasons": incompatible_reasons, "skipped_names": skipped_names, "config": mxfp8_config.CONFIG_NAME, "cache_loaded": cache_loaded}
    finally:
        model.first_stage_model = first_stage
    action = "Loaded cached" if cache_loaded else "Applied"
    print(f"{action} MXFP8 weight quantization for {eligible} policy-allowed Linear modules with coverage {selected_coverage}; policy-skipped {policy_skipped_linear}, technically incompatible {incompatible_linear} ({skipped_reasons})", flush=True)
    timer.record("load mxfp8 cache" if cache_loaded else "apply mxfp8")



def nvfp4_selected_linear_coverage():
    selected = getattr(shared.opts, "nvfp4_linear_coverage", None)
    if selected is None:
        selected = nvfp4_config.LINEAR_COVERAGE_DEFAULT
    if isinstance(selected, str):
        selected = [selected]

    valid = set(nvfp4_config.LINEAR_COVERAGE_CHOICES)
    return {item for item in selected if item in valid}


def nvfp4_linear_region(fqn):
    if fqn.startswith("first_stage_model."):
        return "vae"
    if fqn.startswith("conditioner.") or fqn.startswith("cond_stage_model."):
        return nvfp4_config.LINEAR_COVERAGE_CONDITIONER
    if ".attn1." in fqn:
        return nvfp4_config.LINEAR_COVERAGE_SELF_ATTENTION
    if ".attn2." in fqn:
        return nvfp4_config.LINEAR_COVERAGE_CROSS_ATTENTION
    if fqn.startswith("model.diffusion_model."):
        return nvfp4_config.LINEAR_COVERAGE_UNET_OTHER
    return "other"


def nvfp4_linear_policy_skip_reason(module, fqn):
    region = nvfp4_linear_region(fqn)
    if region in ("vae", "other"):
        return region
    if region not in nvfp4_selected_linear_coverage():
        return region
    return None


def nvfp4_linear_skip_reason(module, fqn):
    if not isinstance(module, torch.nn.Linear):
        return "not_linear"
    policy_reason = nvfp4_linear_policy_skip_reason(module, fqn)
    if policy_reason is not None:
        return policy_reason

    # The A1111 LoRA hook for torch.nn.MultiheadAttention mutates the parent
    # module in_proj_weight/out_proj.weight directly instead of flowing through
    # Linear.forward. When out_proj.weight is an NVFP4Tensor, the backup path
    # attempts weight.to(cpu, copy=True), which TorchAO NVFP4Tensor rejects. Leave
    # those out_proj linears BF16 so active LoRAs remain safe; ordinary Linear
    # layers still use the NVFP4 merge-then-quantize path.
    if fqn.endswith((".attn.out_proj", ".self_attn.out_proj")):
        return "multihead_attention_out_proj_lora_backup"

    return nvfp4_config.technical_linear_skip_reason(module)


def nvfp4_linear_filter(module, fqn):
    return nvfp4_linear_skip_reason(module, fqn) is None


def apply_nvfp4_weight_quantization(model, timer, source_path=None):
    if devices.dtype != torch.bfloat16:
        raise RuntimeError("NVFP4 weight requires --dtype bfloat16; TorchAO NVFP4 kernels do not support float16 activations")
    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 weight requires CUDA")

    first_stage = model.first_stage_model
    model.first_stage_model = None
    eligible = 0
    technical_compatible_linear = 0
    policy_skipped_linear = 0
    incompatible_linear = 0
    skipped_linear = 0
    skipped_reasons = {}
    policy_skipped_reasons = {}
    incompatible_reasons = {}
    skipped_names = []
    managed_modules = []
    managed_ids = set()
    for fqn, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            technical_reason = nvfp4_config.technical_linear_skip_reason(module)
            policy_reason = nvfp4_linear_policy_skip_reason(module, fqn)
            if technical_reason is None:
                technical_compatible_linear += 1
            if policy_reason is not None:
                policy_skipped_linear += 1
                policy_skipped_reasons[policy_reason] = policy_skipped_reasons.get(policy_reason, 0) + 1
                reason = policy_reason
            elif technical_reason is not None:
                incompatible_linear += 1
                incompatible_reasons[technical_reason] = incompatible_reasons.get(technical_reason, 0) + 1
                reason = technical_reason
            else:
                reason = nvfp4_linear_skip_reason(module, fqn)

            if reason is None:
                eligible += 1
                managed_modules.append((fqn, module))
                managed_ids.add(id(module))
            else:
                skipped_linear += 1
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                skipped_names.append({"name": fqn, "reason": reason, "shape": tuple(module.weight.shape) if module.weight is not None else None})
    nvfp4_filter = lambda module, fqn: id(module) in managed_ids
    cache_loaded = False
    try:
        try:
            delattr(model, "network_nvfp4_managed_modules")
        except Exception:
            pass
        for _fqn, module in managed_modules:
            module.network_nvfp4_base_weight = module.weight.detach().to(devices.cpu, copy=True)
            if module.bias is not None:
                module.network_nvfp4_base_bias = module.bias.detach().to(devices.cpu, copy=True)
            else:
                module.network_nvfp4_base_bias = None

        for attr in ("network_nvfp4_active_config_signature", "network_nvfp4_prepare_stats", "network_nvfp4_prepare_error", "network_nvfp4_active_config_ready"):
            try:
                delattr(model, attr)
            except Exception:
                pass

        selected_coverage = sorted(nvfp4_selected_linear_coverage())
        cache_loaded = nvfp4_model_cache.load_into_model(model, source_path, nvfp4_filter, shared.device, selected_coverage)
        if not cache_loaded:
            from torchao.quantization import quantize_
            config = nvfp4_config.get_nvfp4_config()
            nvfp4_config.validate_config(config)
            quantize_(model, config, filter_fn=nvfp4_filter, device=devices.device)
            nvfp4_model_cache.save_from_model(model, source_path, nvfp4_filter, eligible, skipped_linear, skipped_reasons, selected_coverage)
        model.nvfp4_quantization_stats = {"eligible_linear": eligible, "technical_compatible_linear": technical_compatible_linear, "policy_allowed_linear": eligible, "selected_linear_coverage": selected_coverage, "policy_skipped_linear": policy_skipped_linear, "incompatible_linear": incompatible_linear, "skipped_linear": skipped_linear, "skipped_reasons": skipped_reasons, "policy_skipped_reasons": policy_skipped_reasons, "incompatible_reasons": incompatible_reasons, "skipped_names": skipped_names, "config": nvfp4_config.CONFIG_NAME, "cache_loaded": cache_loaded}
    finally:
        model.first_stage_model = first_stage
    action = "Loaded cached" if cache_loaded else "Applied"
    print(f"{action} NVFP4 weight quantization for {eligible} policy-allowed Linear modules with coverage {selected_coverage}; policy-skipped {policy_skipped_linear}, technically incompatible {incompatible_linear} ({skipped_reasons})", flush=True)
    timer.record("load nvfp4 cache" if cache_loaded else "apply nvfp4")


def set_model_type(model, state_dict):
    model.is_sd1 = False
    model.is_sd2 = False
    model.is_sdxl = False
    model.is_ssd = False
    model.is_sd3 = False

    if "model.diffusion_model.x_embedder.proj.weight" in state_dict:
        model.is_sd3 = True
        model.model_type = ModelType.SD3
    elif hasattr(model, 'conditioner'):
        model.is_sdxl = True

        if 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight' not in state_dict.keys():
            model.is_ssd = True
            model.model_type = ModelType.SSD
        else:
            model.model_type = ModelType.SDXL
    elif hasattr(model.cond_stage_model, 'model'):
        model.is_sd2 = True
        model.model_type = ModelType.SD2
    else:
        model.is_sd1 = True
        model.model_type = ModelType.SD1


def set_model_fields(model):
    if not hasattr(model, 'latent_channels'):
        model.latent_channels = 4


def remap_sdxl_clip_text_model_state_dict_if_needed(model, state_dict):
    if not getattr(model, 'is_sdxl', False) or not hasattr(model, 'conditioner'):
        return

    remapped = 0
    for embedder_index, embedder in enumerate(getattr(model.conditioner, 'embedders', [])):
        transformer = getattr(embedder, 'transformer', None)
        if transformer is None or hasattr(transformer, 'text_model'):
            continue

        prefix = f'conditioner.embedders.{embedder_index}.transformer.text_model.'
        for key in [k for k in list(state_dict.keys()) if k.startswith(prefix)]:
            new_key = f'conditioner.embedders.{embedder_index}.transformer.{key[len(prefix):]}'
            state_dict[new_key] = state_dict.pop(key)
            remapped += 1

    if remapped > 0:
        print(f'Remapped {remapped} SDXL CLIP checkpoint keys for flat CLIPTextModel layout')


def load_model_weights(model, checkpoint_info: CheckpointInfo, state_dict, timer):
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    if devices.fp8:
        # prevent model to load state dict in fp8
        model.half()

    if not SkipWritingToConfig.skip:
        shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title

    if state_dict is None:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    set_model_type(model, state_dict)
    set_model_fields(model)
    if 'ztsnr' in state_dict:
        model.ztsnr = True
    else:
        model.ztsnr = False

    if model.is_sdxl:
        sd_models_xl.extend_sdxl(model)

    if model.is_ssd:
        sd_hijack.model_hijack.convert_sdxl_to_ssd(model)

    remap_sdxl_clip_text_model_state_dict_if_needed(model, state_dict)

    if shared.opts.sd_checkpoint_cache > 0 and not check_mxfp8(model) and not check_nvfp4(model):
        # cache newly loaded non-TorchAO-quantized model. MXFP8/NVFP4 reloads
        # need a pristine state_dict because LoadStateDictOnMeta intentionally
        # mutates its input and stale/meta cache entries can later fail with
        # "Cannot copy out of meta tensor; no data!".
        checkpoints_loaded[checkpoint_info] = state_dict.copy()
    elif check_mxfp8(model) or check_nvfp4(model):
        # TorchAO quantized paths must never retain checkpoint state-dict cache
        # entries: the optimized/meta loading path can mutate cached tensors
        # into meta placeholders, and later reloads need pristine disk reads.
        checkpoints_loaded.clear()

    check_weight_quantization_mutual_exclusion(model)

    if hasattr(model, "before_load_weights"):
        model.before_load_weights(state_dict)

    model.load_state_dict(state_dict, strict=False)
    timer.record("apply weights to model")

    if hasattr(model, "after_load_weights"):
        model.after_load_weights(state_dict)

    del state_dict

    # Set is_sdxl_inpaint flag.
    # Checks Unet structure to detect inpaint model. The inpaint model's
    # checkpoint state_dict does not contain the key
    # 'diffusion_model.input_blocks.0.0.weight'.
    diffusion_model_input = model.model.state_dict().get(
        'diffusion_model.input_blocks.0.0.weight'
    )
    model.is_sdxl_inpaint = (
        model.is_sdxl and
        diffusion_model_input is not None and
        diffusion_model_input.shape[1] == 9
    )

    if shared.cmd_opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)
        timer.record("apply channels_last")

    if devices.dtype == torch.float32:
        model.float()
        model.alphas_cumprod_original = model.alphas_cumprod
        devices.dtype_unet = torch.float32
        assert shared.cmd_opts.precision != "half", "Cannot use --precision half with --dtype float32/--no-half"
        timer.record("apply float()")
    else:
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)

        # remove VAE from model-wide dtype conversion; VAE policy is applied explicitly below
        model.first_stage_model = None
        # with --upcast-sampling, don't convert the depth model weights to the low-precision UNet dtype
        if shared.cmd_opts.upcast_sampling and depth_model:
            model.depth_model = None

        alphas_cumprod = model.alphas_cumprod
        model.alphas_cumprod = None
        model.to(devices.dtype)
        model.alphas_cumprod = alphas_cumprod
        model.alphas_cumprod_original = alphas_cumprod
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model

        devices.dtype_unet = devices.dtype
        timer.record(f"apply {devices.dtype} to model")

    apply_alpha_schedule_override(model)

    for module in model.modules():
        if hasattr(module, 'fp16_weight'):
            del module.fp16_weight
        if hasattr(module, 'fp16_bias'):
            del module.fp16_bias

    if check_fp8(model):
        devices.fp8 = True
        first_stage = model.first_stage_model
        model.first_stage_model = None
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if shared.opts.cache_fp16_weight:
                    module.fp16_weight = module.weight.data.clone().cpu().half()
                    if module.bias is not None:
                        module.fp16_bias = module.bias.data.clone().cpu().half()
                module.to(torch.float8_e4m3fn)
        model.first_stage_model = first_stage
        timer.record("apply fp8")
    else:
        devices.fp8 = False

    if check_mxfp8(model):
        devices.mxfp8 = True
        apply_mxfp8_weight_quantization(model, timer, checkpoint_info.filename)
    else:
        devices.mxfp8 = False

    if check_nvfp4(model):
        devices.nvfp4 = True
        apply_nvfp4_weight_quantization(model, timer, checkpoint_info.filename)
    else:
        devices.nvfp4 = False

    devices.unet_needs_upcast = shared.cmd_opts.upcast_sampling and devices.dtype_unet in (torch.float16, torch.bfloat16)

    devices.dtype_vae = torch.float32 if shared.cmd_opts.no_half_vae else devices.dtype
    model.first_stage_model.to(devices.dtype_vae)
    timer.record("apply dtype to VAE")

    # clean up cache if limit is reached
    while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
        checkpoints_loaded.popitem(last=False)

    model.sd_model_hash = sd_model_hash
    model.sd_model_checkpoint = checkpoint_info.filename
    model.sd_checkpoint_info = checkpoint_info
    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    if hasattr(model, 'logvar'):
        model.logvar = model.logvar.to(devices.device)  # fix for training

    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()
    vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename).tuple()
    sd_vae.load_vae(model, vae_file, vae_source)
    timer.record("load VAE")


def enable_midas_autodownload():
    """
    Gives the ldm.modules.midas.api.load_model function automatic downloading.

    When the 512-depth-ema model, and other future models like it, is loaded,
    it calls midas.api.load_model to load the associated midas depth model.
    This function applies a wrapper to download the model to the correct
    location automatically.
    """

    midas_path = os.path.join(paths.models_path, 'midas')

    # stable-diffusion-stability-ai hard-codes the midas model path to
    # a location that differs from where other scripts using this model look.
    # HACK: Overriding the path here.
    for k, v in midas.api.ISL_PATHS.items():
        file_name = os.path.basename(v)
        midas.api.ISL_PATHS[k] = os.path.join(midas_path, file_name)

    midas_urls = {
        "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
        "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
    }

    midas.api.load_model_inner = midas.api.load_model

    def load_model_wrapper(model_type):
        path = midas.api.ISL_PATHS[model_type]
        if not os.path.exists(path):
            if not os.path.exists(midas_path):
                os.mkdir(midas_path)

            print(f"Downloading midas model weights for {model_type} to {path}")
            request.urlretrieve(midas_urls[model_type], path)
            print(f"{model_type} downloaded")

        return midas.api.load_model_inner(model_type)

    midas.api.load_model = load_model_wrapper


def patch_given_betas():
    import ldm.models.diffusion.ddpm

    def patched_register_schedule(*args, **kwargs):
        """a modified version of register_schedule function that converts plain list from Omegaconf into numpy"""

        if isinstance(args[1], ListConfig):
            args = (args[0], np.array(args[1]), *args[2:])

        original_register_schedule(*args, **kwargs)

    original_register_schedule = patches.patch(__name__, ldm.models.diffusion.ddpm.DDPM, 'register_schedule', patched_register_schedule)


def repair_config(sd_config, state_dict=None):
    if not hasattr(sd_config.model.params, "use_ema"):
        sd_config.model.params.use_ema = False

    if hasattr(sd_config.model.params, 'unet_config'):
        if devices.dtype == torch.float32:
            sd_config.model.params.unet_config.params.use_fp16 = False
        elif devices.dtype == torch.float16 and (shared.cmd_opts.upcast_sampling or shared.cmd_opts.precision == "half"):
            sd_config.model.params.unet_config.params.use_fp16 = True

    if hasattr(sd_config.model.params, 'first_stage_config'):
        if getattr(sd_config.model.params.first_stage_config.params.ddconfig, "attn_type", None) == "vanilla-xformers" and not shared.xformers_available:
            sd_config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla"

    # For UnCLIP-L, override the hardcoded karlo directory
    if hasattr(sd_config.model.params, "noise_aug_config") and hasattr(sd_config.model.params.noise_aug_config.params, "clip_stats_path"):
        karlo_path = os.path.join(paths.models_path, 'karlo')
        sd_config.model.params.noise_aug_config.params.clip_stats_path = sd_config.model.params.noise_aug_config.params.clip_stats_path.replace("checkpoints/karlo_models", karlo_path)

    # Do not use checkpoint for inference.
    # This helps prevent extra performance overhead on checking parameters.
    # The perf overhead is about 100ms/it on 4090 for SDXL.
    if hasattr(sd_config.model.params, "network_config"):
        sd_config.model.params.network_config.params.use_checkpoint = False
    if hasattr(sd_config.model.params, "unet_config"):
        sd_config.model.params.unet_config.params.use_checkpoint = False



def rescale_zero_terminal_snr_abar(alphas_cumprod):
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= (alphas_bar_sqrt_T)

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return alphas_bar


def apply_alpha_schedule_override(sd_model, p=None):
    """
    Applies an override to the alpha schedule of the model according to settings.
    - downcasts the alpha schedule to half precision
    - rescales the alpha schedule to have zero terminal SNR
    """

    if not hasattr(sd_model, 'alphas_cumprod') or not hasattr(sd_model, 'alphas_cumprod_original'):
        return

    sd_model.alphas_cumprod = sd_model.alphas_cumprod_original.to(shared.device)

    if opts.use_downcasted_alpha_bar:
        if p is not None:
            p.extra_generation_params['Downcast alphas_cumprod'] = opts.use_downcasted_alpha_bar
        sd_model.alphas_cumprod = sd_model.alphas_cumprod.half().to(shared.device)

    if opts.sd_noise_schedule == "Zero Terminal SNR" or (hasattr(sd_model, 'ztsnr') and sd_model.ztsnr):
        if p is not None:
            p.extra_generation_params['Noise Schedule'] = opts.sd_noise_schedule
        sd_model.alphas_cumprod = rescale_zero_terminal_snr_abar(sd_model.alphas_cumprod).to(shared.device)


sd1_clip_weight = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight = 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'
sdxl_clip_weight = 'conditioner.embedders.1.model.ln_final.weight'
sdxl_refiner_clip_weight = 'conditioner.embedders.0.model.ln_final.weight'


class SdModelData:
    def __init__(self):
        self.sd_model = None
        self.loaded_sd_models = []
        self.was_loaded_at_least_once = False
        self.lock = threading.Lock()

    def get_sd_model(self):
        if self.was_loaded_at_least_once:
            return self.sd_model

        if self.sd_model is None:
            with self.lock:
                if self.sd_model is not None or self.was_loaded_at_least_once:
                    return self.sd_model

                try:
                    load_model()

                except Exception as e:
                    errors.display(e, "loading stable diffusion model", full_traceback=True)
                    print("", file=sys.stderr)
                    print("Stable diffusion model failed to load", file=sys.stderr)
                    self.sd_model = None

        return self.sd_model

    def set_sd_model(self, v, already_loaded=False):
        self.sd_model = v
        if already_loaded:
            sd_vae.base_vae = getattr(v, "base_vae", None)
            sd_vae.loaded_vae_file = getattr(v, "loaded_vae_file", None)
            sd_vae.checkpoint_info = v.sd_checkpoint_info

        try:
            self.loaded_sd_models.remove(v)
        except ValueError:
            pass

        if v is not None:
            self.loaded_sd_models.insert(0, v)


model_data = SdModelData()


def get_empty_cond(sd_model):

    p = processing.StableDiffusionProcessingTxt2Img()
    extra_networks.activate(p, {})

    if hasattr(sd_model, 'get_learned_conditioning'):
        d = sd_model.get_learned_conditioning([""])
    else:
        d = sd_model.cond_stage_model([""])

    if isinstance(d, dict):
        d = d['crossattn']

    return d


def send_model_to_cpu(m):
    if m is not None:
        if m.lowvram:
            lowvram.send_everything_to_cpu()
        else:
            m.to(devices.cpu)

    devices.torch_gc()


def model_target_device(m):
    if lowvram.is_needed(m):
        return devices.cpu
    else:
        return devices.device


def torchao_quant_tensor_types():
    types = []
    try:
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        types.append(MXTensor)
    except Exception:
        pass
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
        types.append(NVFP4Tensor)
    except Exception:
        pass
    return tuple(types)


def send_torchao_quant_model_to_device(m):
    torchao_tensor_types = torchao_quant_tensor_types()
    target = shared.device
    for module in m.modules():
        for name, param in list(module._parameters.items()):
            if param is None or isinstance(param, torchao_tensor_types):
                continue
            if getattr(param, "device", None) != target:
                module._parameters[name] = torch.nn.Parameter(param.to(target), requires_grad=param.requires_grad)

        for name, buffer in list(module._buffers.items()):
            if buffer is None or isinstance(buffer, torchao_tensor_types):
                continue
            if getattr(buffer, "device", None) != target:
                module._buffers[name] = buffer.to(target)


def send_model_to_device(m):
    lowvram.apply(m)

    if not m.lowvram:
        if devices.mxfp8 or devices.nvfp4:
            # TorchAO tensor subclasses do not implement all tensor-moving /
            # aliasing ops that nn.Module.to() may call. Move ordinary
            # parameters/buffers around quantized leaves instead so skipped
            # BF16 regions are not stranded on CPU.
            send_torchao_quant_model_to_device(m)
            return
        m.to(shared.device)


def send_model_to_trash(m):
    m.to(device="meta")
    devices.torch_gc()


def instantiate_from_config(config, state_dict=None):
    constructor = get_obj_from_str(config["target"])

    params = {**config.get("params", {})}

    if state_dict and "state_dict" in params and params["state_dict"] is None:
        params["state_dict"] = state_dict

    return constructor(**params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model(checkpoint_info=None, already_loaded_state_dict=None, checkpoint_config=None):
    from modules import sd_hijack
    checkpoint_info = checkpoint_info or select_checkpoint()

    timer = Timer()

    if model_data.sd_model:
        send_model_to_trash(model_data.sd_model)
        model_data.sd_model = None
        devices.torch_gc()

    timer.record("unload existing model")

    if already_loaded_state_dict is not None:
        state_dict = already_loaded_state_dict
    else:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    if not checkpoint_config:
        checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    clip_is_included_into_sd = any(x for x in [sd1_clip_weight, sd2_clip_weight, sdxl_clip_weight, sdxl_refiner_clip_weight] if x in state_dict)

    timer.record("find config")

    sd_config = OmegaConf.load(checkpoint_config)
    repair_config(sd_config, state_dict)

    timer.record("load config")

    print(f"Creating model from config: {checkpoint_config}")

    sd_model = None
    try:
        with sd_disable_initialization.DisableInitialization(disable_clip=clip_is_included_into_sd or shared.cmd_opts.do_not_download_clip):
            with DisableFastModelLoadingForTorchAOQuant():
                with sd_disable_initialization.InitializeOnMeta():
                    sd_model = instantiate_from_config(sd_config.model, state_dict)

    except Exception as e:
        errors.display(e, "creating model quickly", full_traceback=True)

    if sd_model is None:
        print('Failed to create model quickly; will retry using slow method.', file=sys.stderr)

        with DisableFastModelLoadingForTorchAOQuant():
            with sd_disable_initialization.InitializeOnMeta():
                sd_model = instantiate_from_config(sd_config.model, state_dict)

    sd_model.used_config = checkpoint_config

    timer.record("create model")

    if devices.dtype == torch.float32:
        weight_dtype_conversion = None
    else:
        weight_dtype_conversion = {
            'first_stage_model': None,
            'alphas_cumprod': None,
            '': devices.dtype,
        }

    # TorchAO quantized load/reload correctness matters more than the generic
    # meta-device RAM optimization: the optimized path can strand meta placeholders when
    # custom SDXL/OpenCLIP/VAE loaders bypass the patched Module path, producing
    # "Cannot copy out of meta tensor; no data!" on later reloads.
    with DisableFastModelLoadingForTorchAOQuant():
        with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device=model_target_device(sd_model), weight_dtype_conversion=weight_dtype_conversion):
            load_model_weights(sd_model, checkpoint_info, state_dict, timer)

    timer.record("load weights from state dict")

    send_model_to_device(sd_model)
    timer.record("move model to device")

    sd_hijack.model_hijack.hijack(sd_model)

    timer.record("hijack")

    sd_model.eval()
    model_data.set_sd_model(sd_model)
    model_data.was_loaded_at_least_once = True

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)  # Reload embeddings after model load as they may or may not fit the model

    timer.record("load textual inversion embeddings")

    script_callbacks.model_loaded_callback(sd_model)

    timer.record("scripts callbacks")

    with devices.autocast(), torch.no_grad():
        sd_model.cond_stage_model_empty_prompt = get_empty_cond(sd_model)

    timer.record("calculate empty prompt")

    print(f"Model loaded in {timer.summary()}.")

    return sd_model


def reuse_model_from_already_loaded(sd_model, checkpoint_info, timer):
    """
    Checks if the desired checkpoint from checkpoint_info is not already loaded in model_data.loaded_sd_models.
    If it is loaded, returns that (moving it to GPU if necessary, and moving the currently loadded model to CPU if necessary).
    If not, returns the model that can be used to load weights from checkpoint_info's file.
    If no such model exists, returns None.
    Additionally deletes loaded models that are over the limit set in settings (sd_checkpoints_limit).
    """

    if sd_model is not None and sd_model.sd_checkpoint_info.filename == checkpoint_info.filename:
        return sd_model

    if shared.opts.sd_checkpoints_keep_in_cpu:
        send_model_to_cpu(sd_model)
        timer.record("send model to cpu")

    already_loaded = None
    for i in reversed(range(len(model_data.loaded_sd_models))):
        loaded_model = model_data.loaded_sd_models[i]
        if loaded_model.sd_checkpoint_info.filename == checkpoint_info.filename:
            already_loaded = loaded_model
            continue

        if len(model_data.loaded_sd_models) > shared.opts.sd_checkpoints_limit > 0:
            print(f"Unloading model {len(model_data.loaded_sd_models)} over the limit of {shared.opts.sd_checkpoints_limit}: {loaded_model.sd_checkpoint_info.title}")
            del model_data.loaded_sd_models[i]
            send_model_to_trash(loaded_model)
            timer.record("send model to trash")

    if already_loaded is not None:
        send_model_to_device(already_loaded)
        timer.record("send model to device")

        model_data.set_sd_model(already_loaded, already_loaded=True)

        if not SkipWritingToConfig.skip:
            shared.opts.data["sd_model_checkpoint"] = already_loaded.sd_checkpoint_info.title
            shared.opts.data["sd_checkpoint_hash"] = already_loaded.sd_checkpoint_info.sha256

        print(f"Using already loaded model {already_loaded.sd_checkpoint_info.title}: done in {timer.summary()}")
        sd_vae.reload_vae_weights(already_loaded)
        return model_data.sd_model
    elif shared.opts.sd_checkpoints_limit > 1 and len(model_data.loaded_sd_models) < shared.opts.sd_checkpoints_limit:
        print(f"Loading model {checkpoint_info.title} ({len(model_data.loaded_sd_models) + 1} out of {shared.opts.sd_checkpoints_limit})")

        model_data.sd_model = None
        load_model(checkpoint_info)
        return model_data.sd_model
    elif len(model_data.loaded_sd_models) > 0:
        sd_model = model_data.loaded_sd_models.pop()
        model_data.sd_model = sd_model

        sd_vae.base_vae = getattr(sd_model, "base_vae", None)
        sd_vae.loaded_vae_file = getattr(sd_model, "loaded_vae_file", None)
        sd_vae.checkpoint_info = sd_model.sd_checkpoint_info

        print(f"Reusing loaded model {sd_model.sd_checkpoint_info.title} to load {checkpoint_info.title}")
        return sd_model
    else:
        return None


def reload_model_weights(sd_model=None, info=None, forced_reload=False):
    checkpoint_info = info or select_checkpoint()

    timer = Timer()

    if not sd_model:
        sd_model = model_data.sd_model

    torchao_quant_mode_changed = False
    if sd_model is None:  # previous model load failed
        current_checkpoint_info = None
    else:
        current_checkpoint_info = sd_model.sd_checkpoint_info
        if check_fp8(sd_model) != devices.fp8:
            # load from state dict again to prevent extra numerical errors
            forced_reload = True
        elif check_mxfp8(sd_model) != devices.mxfp8:
            # MXTensor parameters cannot safely be overwritten by the normal
            # state_dict reload path. Switching MXFP8 on/off needs a fresh
            # model instance rather than copying checkpoint tensors into the
            # existing MXFP8-mutated module tree.
            forced_reload = True
            torchao_quant_mode_changed = True
        elif check_nvfp4(sd_model) != devices.nvfp4:
            # NVFP4Tensor parameters cannot safely be overwritten by the normal
            # state_dict reload path. Switching NVFP4 on/off needs a fresh
            # model instance rather than copying checkpoint tensors into the
            # existing NVFP4-mutated module tree.
            forced_reload = True
            torchao_quant_mode_changed = True
        elif devices.mxfp8 and sorted(getattr(sd_model, "mxfp8_quantization_stats", {}).get("selected_linear_coverage", [])) != sorted(mxfp8_selected_linear_coverage()):
            # Changing Linear coverage can move already-quantized MXTensor
            # parameters back out of the active policy. Reuse/reload would try
            # to copy BF16 checkpoint tensors into the existing MXTensor
            # module tree and can fail, so build a fresh model instance.
            forced_reload = True
            torchao_quant_mode_changed = True
        elif devices.nvfp4 and sorted(getattr(sd_model, "nvfp4_quantization_stats", {}).get("selected_linear_coverage", [])) != sorted(nvfp4_selected_linear_coverage()):
            # Changing Linear coverage can move already-quantized NVFP4Tensor
            # parameters back out of the active policy. Reuse/reload would try
            # to copy BF16 checkpoint tensors into the existing NVFP4Tensor
            # module tree and can fail, so build a fresh model instance.
            forced_reload = True
            torchao_quant_mode_changed = True
        elif forced_reload and ((devices.mxfp8 and check_mxfp8(sd_model)) or (devices.nvfp4 and check_nvfp4(sd_model))):
            # Option onchange hooks pass forced_reload=True even when the selected
            # coverage value resolves to the same effective policy. The normal
            # forced reload path is still unsafe for a TorchAO-mutated tree because
            # it can reuse meta-mutated checkpoint cache tensors and/or copy into
            # TorchAO tensor-backed modules. Treat any forced reload of an active
            # TorchAO quantized model as a fresh reload.
            torchao_quant_mode_changed = True
        elif sd_model.sd_model_checkpoint == checkpoint_info.filename and not forced_reload:
            return sd_model

    if forced_reload and (devices.mxfp8 or devices.nvfp4 or bool(getattr(sd_model, "mxfp8_quantization_stats", None)) or bool(getattr(sd_model, "nvfp4_quantization_stats", None)) or check_mxfp8(sd_model) or check_nvfp4(sd_model)):
        # forced_reload can be set before the TorchAO-quant-specific branches above
        # get a chance to classify the reload, including early option-onchange calls
        # while model_data.sd_model is temporarily unset. Once MXFP8/NVFP4 is active
        # or currently requested, the generic reload path is unsafe; force the fresh
        # uncached TorchAO quantized path.
        torchao_quant_mode_changed = True

    if torchao_quant_mode_changed:
        # LoadStateDictOnMeta mutates the dict it receives and get_checkpoint_state_dict()
        # returns the cached dict by reference. If a cache entry from a previous
        # optimized/meta load survives, a fresh model can still try to materialize
        # parameters from meta checkpoint tensors and fail with
        # "Cannot copy out of meta tensor; no data!". Invalidate every cache key
        # for this checkpoint filename (CheckpointInfo object identity/equality is
        # not a safe enough lookup here) and read directly from disk.
        checkpoints_loaded.clear()
        print(f"Reloading TorchAO-quantized model from uncached checkpoint storage: {checkpoint_info.filename}")
        state_dict = read_state_dict(checkpoint_info.filename)
        timer.record("load weights from disk")
        checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
        timer.record("find config")

        # load_model() normally moves the existing model to meta before
        # replacing it. TorchAO tensor subclasses do not support that path
        # reliably (meta/cuda storage alias correction can throw), so detach
        # the old MXFP8/NVFP4-mutated tree before constructing the fresh model
        # instance.
        old_sd_model = model_data.sd_model
        model_data.sd_model = None
        model_data.loaded_sd_models = [model for model in model_data.loaded_sd_models if model is not old_sd_model]
        if old_sd_model is not None:
            try:
                sd_hijack.model_hijack.undo_hijack(old_sd_model)
            except Exception:
                pass
            del old_sd_model
            devices.torch_gc()

        load_model(checkpoint_info, already_loaded_state_dict=state_dict, checkpoint_config=checkpoint_config)
        return model_data.sd_model

    sd_model = reuse_model_from_already_loaded(sd_model, checkpoint_info, timer)
    if not forced_reload and sd_model is not None and sd_model.sd_checkpoint_info.filename == checkpoint_info.filename:
        return sd_model

    if sd_model is not None:
        sd_unet.apply_unet("None")
        send_model_to_cpu(sd_model)
        sd_hijack.model_hijack.undo_hijack(sd_model)

    state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)

    timer.record("find config")

    if sd_model is None or checkpoint_config != sd_model.used_config:
        if sd_model is not None:
            send_model_to_trash(sd_model)

        load_model(checkpoint_info, already_loaded_state_dict=state_dict, checkpoint_config=checkpoint_config)
        return model_data.sd_model

    try:
        load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    except Exception:
        print("Failed to load checkpoint, restoring previous")
        load_model_weights(sd_model, current_checkpoint_info, None, timer)
        raise
    finally:
        sd_hijack.model_hijack.hijack(sd_model)
        timer.record("hijack")

        if not sd_model.lowvram:
            sd_model.to(devices.device)
            timer.record("move model to device")

        script_callbacks.model_loaded_callback(sd_model)
        timer.record("script callbacks")

    print(f"Weights loaded in {timer.summary()}.")

    model_data.set_sd_model(sd_model)
    sd_unet.apply_unet()

    return sd_model


def unload_model_weights(sd_model=None, info=None):
    send_model_to_cpu(sd_model or shared.sd_model)

    return sd_model


def apply_token_merging(sd_model, token_merging_ratio):
    """
    Applies speed and memory optimizations from tomesd.
    """

    current_token_merging_ratio = getattr(sd_model, 'applied_token_merged_ratio', 0)

    if current_token_merging_ratio == token_merging_ratio:
        return

    if current_token_merging_ratio > 0:
        tomesd.remove_patch(sd_model)

    if token_merging_ratio > 0:
        tomesd.apply_patch(
            sd_model,
            ratio=token_merging_ratio,
            use_rand=False,  # can cause issues with some samplers
            merge_attn=True,
            merge_crossattn=False,
            merge_mlp=False
        )

    sd_model.applied_token_merged_ratio = token_merging_ratio
