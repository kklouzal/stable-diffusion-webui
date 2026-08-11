import os
import collections
import sys
from dataclasses import dataclass

from modules import paths, shared, devices, script_callbacks, sd_models, extra_networks, lowvram, sd_hijack, hashes, openclaw_cuda_graphs, openclaw_lifecycle_epochs

import glob
from copy import deepcopy


vae_path = os.path.abspath(os.path.join(paths.models_path, "VAE"))
vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
vae_dict = {}


base_vae = None
loaded_vae_file = None
checkpoint_info = None

checkpoints_loaded = collections.OrderedDict()


def _vae_cache_key(vae_file):
    filename = os.path.abspath(vae_file)
    try:
        stat = os.stat(filename)
        file_identity = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        file_identity = (None, None)

    return (filename, file_identity)


def _vae_cache_key_filename(cache_key):
    if isinstance(cache_key, tuple) and cache_key:
        return cache_key[0]
    return os.path.abspath(cache_key) if isinstance(cache_key, str) else None


def _drop_stale_vae_cache_entries(vae_file, current_cache_key):
    filename = os.path.abspath(vae_file)
    for cache_key in list(checkpoints_loaded.keys()):
        if cache_key != current_cache_key and _vae_cache_key_filename(cache_key) == filename:
            del checkpoints_loaded[cache_key]


def get_loaded_vae_name():
    if loaded_vae_file is None:
        return None

    return os.path.basename(loaded_vae_file)


def get_loaded_vae_hash():
    if loaded_vae_file is None:
        return None

    sha256 = hashes.sha256(loaded_vae_file, 'vae')

    return sha256[0:10] if sha256 else None


def get_base_vae(model):
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_vae
    return None


def store_base_vae(model):
    global base_vae, checkpoint_info
    if checkpoint_info != model.sd_checkpoint_info:
        assert not loaded_vae_file, "Trying to store non-base VAE!"
        base_vae = deepcopy(model.first_stage_model.state_dict())
        checkpoint_info = model.sd_checkpoint_info


def delete_base_vae():
    global base_vae, checkpoint_info
    base_vae = None
    checkpoint_info = None


def restore_base_vae(model):
    global loaded_vae_file
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info:
        print("Restoring base VAE")
        _load_vae_dict(model, base_vae)
        loaded_vae_file = None
        openclaw_lifecycle_epochs.note_vae_commit(model, bytes_changed=False, object_changed=True, publish=False)
    delete_base_vae()


def get_filename(filepath):
    return os.path.basename(filepath)


def vae_search_paths():
    paths = [
        os.path.join(sd_models.model_path, '**/*.vae.ckpt'),
        os.path.join(sd_models.model_path, '**/*.vae.pt'),
        os.path.join(sd_models.model_path, '**/*.vae.safetensors'),
        os.path.join(vae_path, '**/*.ckpt'),
        os.path.join(vae_path, '**/*.pt'),
        os.path.join(vae_path, '**/*.safetensors'),
    ]

    if shared.cmd_opts.ckpt_dir is not None and os.path.isdir(shared.cmd_opts.ckpt_dir):
        paths += [
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.ckpt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.pt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.safetensors'),
        ]

    if shared.cmd_opts.vae_dir is not None and os.path.isdir(shared.cmd_opts.vae_dir):
        paths += [
            os.path.join(shared.cmd_opts.vae_dir, '**/*.ckpt'),
            os.path.join(shared.cmd_opts.vae_dir, '**/*.pt'),
            os.path.join(shared.cmd_opts.vae_dir, '**/*.safetensors'),
        ]

    return paths


def refresh_vae_list():
    vae_dict.clear()

    candidates = []
    for path in vae_search_paths():
        candidates += glob.iglob(path, recursive=True)

    for filepath in candidates:
        name = get_filename(filepath)
        vae_dict[name] = filepath

    vae_dict.update(dict(sorted(vae_dict.items(), key=lambda item: shared.natural_sort_key(item[0]))))


def find_vae_near_checkpoint(checkpoint_file):
    checkpoint_path = os.path.basename(checkpoint_file).rsplit('.', 1)[0]
    for vae_file in vae_dict.values():
        if os.path.basename(vae_file).startswith(checkpoint_path):
            return vae_file

    return None


@dataclass
class VaeResolution:
    vae: str = None
    source: str = None
    resolved: bool = True

    def tuple(self):
        return self.vae, self.source


def is_automatic():
    return shared.opts.sd_vae in {"Automatic", "auto"}  # "auto" for people with old config


def resolve_vae_from_setting() -> VaeResolution:
    if shared.opts.sd_vae == "None":
        return VaeResolution()

    vae_from_options = vae_dict.get(shared.opts.sd_vae, None)
    if vae_from_options is not None:
        return VaeResolution(vae_from_options, 'specified in settings')

    if not is_automatic():
        print(f"Couldn't find VAE named {shared.opts.sd_vae}; using None instead")

    return VaeResolution(resolved=False)


def resolve_vae_from_user_metadata(checkpoint_file) -> VaeResolution:
    metadata = extra_networks.get_user_metadata(checkpoint_file)
    vae_metadata = metadata.get("vae", None)
    if vae_metadata is not None and vae_metadata != "Automatic":
        if vae_metadata == "None":
            return VaeResolution()

        vae_from_metadata = vae_dict.get(vae_metadata, None)
        if vae_from_metadata is not None:
            return VaeResolution(vae_from_metadata, "from user metadata")

    return VaeResolution(resolved=False)


def resolve_vae_near_checkpoint(checkpoint_file) -> VaeResolution:
    vae_near_checkpoint = find_vae_near_checkpoint(checkpoint_file)
    if vae_near_checkpoint is not None and (not shared.opts.sd_vae_overrides_per_model_preferences or is_automatic()):
        return VaeResolution(vae_near_checkpoint, 'found near the checkpoint')

    return VaeResolution(resolved=False)


def resolve_vae(checkpoint_file) -> VaeResolution:
    if shared.cmd_opts.vae_path is not None:
        return VaeResolution(shared.cmd_opts.vae_path, 'from commandline argument')

    if shared.opts.sd_vae_overrides_per_model_preferences and not is_automatic():
        return resolve_vae_from_setting()

    res = resolve_vae_from_user_metadata(checkpoint_file)
    if res.resolved:
        return res

    res = resolve_vae_near_checkpoint(checkpoint_file)
    if res.resolved:
        return res

    res = resolve_vae_from_setting()

    return res


def load_vae_dict(filename, map_location):
    vae_ckpt = sd_models.read_state_dict(filename, map_location=map_location)
    vae_dict_1 = {k: v for k, v in vae_ckpt.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    return vae_dict_1


def load_vae(model, vae_file=None, vae_source="from unknown source"):
    global vae_dict, base_vae, loaded_vae_file
    # save_settings = False

    cache_enabled = shared.opts.sd_vae_checkpoint_cache > 0

    if vae_file:
        vae_cache_key = _vae_cache_key(vae_file)
        _drop_stale_vae_cache_entries(vae_file, vae_cache_key)
        if cache_enabled and vae_cache_key in checkpoints_loaded:
            # use vae checkpoint cache
            print(f"Loading VAE weights {vae_source}: cached {get_filename(vae_file)}")
            store_base_vae(model)
            _load_vae_dict(model, checkpoints_loaded[vae_cache_key])
        else:
            assert os.path.isfile(vae_file), f"VAE {vae_source} doesn't exist: {vae_file}"
            print(f"Loading VAE weights {vae_source}: {vae_file}")
            store_base_vae(model)

            vae_dict_1 = load_vae_dict(vae_file, map_location=shared.weight_load_location)
            _load_vae_dict(model, vae_dict_1)

            if cache_enabled:
                # cache newly loaded vae
                checkpoints_loaded[vae_cache_key] = vae_dict_1.copy()

        # clean up cache if limit is reached
        if cache_enabled:
            while len(checkpoints_loaded) > shared.opts.sd_vae_checkpoint_cache + 1: # we need to count the current model
                checkpoints_loaded.popitem(last=False)  # LRU

        # If vae used is not in dict, update it
        # It will be removed on refresh though
        vae_opt = get_filename(vae_file)
        if vae_opt not in vae_dict:
            vae_dict[vae_opt] = vae_file

    elif loaded_vae_file:
        restore_base_vae(model)

    loaded_vae_file = vae_file
    model.base_vae = base_vae
    model.loaded_vae_file = loaded_vae_file
    openclaw_cuda_graphs.note_vae_loaded(model, "vae_changed")
    openclaw_lifecycle_epochs.note_vae_commit(model, bytes_changed=vae_file is not None, object_changed=True, publish=False)
    openclaw_lifecycle_epochs.note_vae_commit(model, bytes_changed=vae_file is not None, object_changed=True, publish=False)


# don't call this from outside
def _load_vae_dict(model, vae_dict_1):
    model.first_stage_model.load_state_dict(vae_dict_1)
    model.first_stage_model.to(devices.dtype_vae)


def clear_loaded_vae():
    global loaded_vae_file
    loaded_vae_file = None


unspecified = object()


def reload_vae_weights(sd_model=None, vae_file=unspecified):
    if not sd_model:
        sd_model = shared.sd_model

    checkpoint_info = sd_model.sd_checkpoint_info
    checkpoint_file = checkpoint_info.filename

    if vae_file == unspecified:
        vae_file, vae_source = resolve_vae(checkpoint_file).tuple()
    else:
        vae_source = "from function argument"

    if loaded_vae_file == vae_file:
        return

    reload_exc_info = None
    needs_finalization = False
    try:
        if sd_model.lowvram:
            lowvram.send_everything_to_cpu()
        else:
            sd_models.send_model_to_cpu(sd_model)

        sd_hijack.model_hijack.undo_hijack(sd_model)
        needs_finalization = True

        try:
            load_vae(sd_model, vae_file, vae_source)
        except Exception:
            reload_exc_info = sys.exc_info()
            raise
    finally:
        if needs_finalization:
            try:
                sd_hijack.model_hijack.hijack(sd_model)

                if not sd_model.lowvram:
                    sd_models.send_model_to_device(sd_model)

                script_callbacks.model_loaded_callback(sd_model)
            except Exception as finalization_exception:
                if reload_exc_info is not None:
                    print(f"Failed to finalize model after VAE reload failure; preserving original exception: {finalization_exception}", flush=True)
                    raise reload_exc_info[1].with_traceback(reload_exc_info[2]) from finalization_exception
                raise

    vae_bytes_changed, vae_object_changed = openclaw_lifecycle_epochs.take_pending_vae_commit(sd_model)
    openclaw_lifecycle_epochs.note_vae_commit(sd_model, bytes_changed=vae_bytes_changed, object_changed=vae_object_changed, publish=True)
    print("VAE weights loaded.")
    return sd_model
