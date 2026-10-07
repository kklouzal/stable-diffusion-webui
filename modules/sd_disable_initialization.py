import inspect
import logging
import threading

import ldm.modules.encoders.modules
import open_clip
import torch
import transformers.utils.hub

from modules import shared

# The replacements below are installed on shared objects (torch.nn classes, torch.nn.init, open_clip, transformers),
# but a model load runs on one thread while API handlers and the startup model-load thread run on others. So a
# replacement acts only on the thread that entered the `with` block and calls the original on every other thread, and
# one process-wide re-entrant lock is held from entry to exit: blocks on different threads never overlap, so each block
# restores exactly what it found (blocks nested on one thread stay LIFO through `with`). Code inside a block must not
# wait for another thread that enters one of these blocks.
_replacement_lock = threading.RLock()
_MISSING = object()


class ReplaceHelper:
    """Base for context managers that swap attributes for the duration of a `with` block, on the entering thread only.

    Subclasses install their replacements with `replace()` from `install()`, and may skip the block by returning False
    from `enabled()`. Leaving the block, or a failure while installing, restores every replaced attribute: a value the
    object held itself is set back, and an inherited one is deleted so lookup reaches the base class again.
    """

    def __init__(self):
        self.replaced = []
        self.owner = None

    def enabled(self):
        return True

    def install(self):
        raise NotImplementedError

    def __enter__(self):
        if not self.enabled():
            return self

        _replacement_lock.acquire()
        self.owner = threading.get_ident()
        try:
            self.install()
        except BaseException:
            self.restore()
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()

    def replace(self, obj, field, func):
        """Make `obj.field` call `func` on this block's thread; return the original, or None (and replace nothing) if absent."""
        if self.owner != threading.get_ident():
            raise RuntimeError("ReplaceHelper.replace() is only valid inside its own `with` block")

        original = getattr(obj, field, None)
        if original is None:
            return None

        owner = self.owner
        descriptor = inspect.getattr_static(obj, field, None)
        if isinstance(descriptor, classmethod):
            def replacement(cls, *args, **kwargs):
                if threading.get_ident() == owner:
                    return func(*args, **kwargs)
                return descriptor.__get__(None, cls)(*args, **kwargs)

            replacement = classmethod(replacement)
        else:
            def replacement(*args, **kwargs):
                if threading.get_ident() == owner:
                    return func(*args, **kwargs)
                return original(*args, **kwargs)

            if isinstance(descriptor, staticmethod):
                replacement = staticmethod(replacement)

        self.replaced.append((obj, field, vars(obj).get(field, _MISSING)))
        setattr(obj, field, replacement)

        return original

    def restore(self):
        if self.owner is None:
            return

        try:
            for obj, field, own_value in reversed(self.replaced):
                if own_value is _MISSING:
                    delattr(obj, field)
                else:
                    setattr(obj, field, own_value)
        finally:
            self.replaced.clear()
            self.owner = None
            _replacement_lock.release()


class DisableInitialization(ReplaceHelper):
    """
    When an object of this class enters a `with` block, it starts:
    - preventing torch's layer initialization functions from working
    - changes CLIP and OpenCLIP to not download model weights
    - changes CLIP to not make requests to check if there is a new version of a file you already have

    When it leaves the block, it reverts everything to how it was before. Only the thread that entered the block is
    affected.

    Use it like this:
    ```
    with DisableInitialization():
        do_things()
    ```
    """

    def __init__(self, disable_clip=True):
        super().__init__()
        self.disable_clip = disable_clip

    def install(self):
        def skip_initialization(tensor, *args, **kwargs):
            return tensor

        def create_model_and_transforms_without_pretrained(*args, pretrained=None, **kwargs):
            class SuppressNoPretrainedOpenClipWarning(logging.Filter):
                def filter(self, record):
                    return not (
                        record.levelno == logging.WARNING
                        and isinstance(record.getMessage(), str)
                        and record.getMessage().startswith("No pretrained weights loaded for model ")
                        and record.getMessage().endswith(". Model initialized randomly.")
                    )

            root_logger = logging.getLogger()
            warning_filter = SuppressNoPretrainedOpenClipWarning()
            root_logger.addFilter(warning_filter)
            try:
                return self.create_model_and_transforms(*args, pretrained=None, **kwargs)
            finally:
                root_logger.removeFilter(warning_filter)

        def CLIPTextModel_from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
            res = self.CLIPTextModel_from_pretrained(None, *model_args, config=pretrained_model_name_or_path, state_dict={}, **kwargs)
            res.name_or_path = pretrained_model_name_or_path
            return res

        def transformers_utils_hub_get_file_from_cache(original, url, *args, local_files_only=False, **kwargs):
            filename = args[0] if args else kwargs.get("filename")

            # this file is always 404, prevent making request
            if url == f'{shared.hf_endpoint}/openai/clip-vit-large-patch14/resolve/main/added_tokens.json' or url == 'openai/clip-vit-large-patch14' and filename == 'added_tokens.json':
                return None

            try:
                res = original(url, *args, local_files_only=True, **kwargs)
                if res is None and not local_files_only:
                    res = original(url, *args, local_files_only=False, **kwargs)
                return res
            except Exception:
                if local_files_only:
                    raise
                return original(url, *args, local_files_only=False, **kwargs)

        def transformers_tokenization_utils_base_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_tokenization_utils_base_cached_file, url, *args, local_files_only=local_files_only, **kwargs)

        def transformers_configuration_utils_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_configuration_utils_cached_file, url, *args, local_files_only=local_files_only, **kwargs)

        self.replace(torch.nn.init, 'kaiming_uniform_', skip_initialization)
        self.replace(torch.nn.init, '_no_grad_normal_', skip_initialization)
        self.replace(torch.nn.init, '_no_grad_uniform_', skip_initialization)

        if self.disable_clip:
            self.create_model_and_transforms = self.replace(open_clip, 'create_model_and_transforms', create_model_and_transforms_without_pretrained)
            self.CLIPTextModel_from_pretrained = self.replace(ldm.modules.encoders.modules.CLIPTextModel, 'from_pretrained', CLIPTextModel_from_pretrained)
            self.transformers_tokenization_utils_base_cached_file = self.replace(transformers.tokenization_utils_base, 'cached_file', transformers_tokenization_utils_base_cached_file)
            self.transformers_configuration_utils_cached_file = self.replace(transformers.configuration_utils, 'cached_file', transformers_configuration_utils_cached_file)


class InitializeOnMeta(ReplaceHelper):
    """
    Context manager that causes all parameters for linear/conv2d/mha layers to be allocated on meta device,
    which results in those parameters having no values and taking no memory. model.to() will be broken and
    will need to be repaired by using LoadStateDictOnMeta below when loading params from state dict.
    Only the thread that entered the block is affected.

    Usage:
    ```
    with sd_disable_initialization.InitializeOnMeta():
        sd_model = instantiate_from_config(sd_config.model)
    ```
    """

    def enabled(self):
        return not shared.cmd_opts.disable_model_loading_ram_optimization

    def install(self):
        def set_device(x):
            x["device"] = "meta"
            return x

        linear_init = self.replace(torch.nn.Linear, '__init__', lambda *args, **kwargs: linear_init(*args, **set_device(kwargs)))
        conv2d_init = self.replace(torch.nn.Conv2d, '__init__', lambda *args, **kwargs: conv2d_init(*args, **set_device(kwargs)))
        mha_init = self.replace(torch.nn.MultiheadAttention, '__init__', lambda *args, **kwargs: mha_init(*args, **set_device(kwargs)))
        self.replace(torch.nn.Module, 'to', lambda module, *args, **kwargs: module)


class LoadStateDictOnMeta(ReplaceHelper):
    """
    Context manager that allows to read parameters from state_dict into a model that has some of its parameters in the meta device.
    As those parameters are read from state_dict, they will be deleted from it, so by the end state_dict will be mostly empty, to save memory.
    Meant to be used together with InitializeOnMeta above. Only the thread that entered the block is affected.

    Usage:
    ```
    with sd_disable_initialization.LoadStateDictOnMeta(state_dict):
        model.load_state_dict(state_dict, strict=False)
    ```
    """

    def __init__(self, state_dict, device, weight_dtype_conversion=None):
        super().__init__()
        self.state_dict = state_dict
        self.device = device
        self.weight_dtype_conversion = weight_dtype_conversion or {}
        self.default_dtype = self.weight_dtype_conversion.get('')

    def get_weight_dtype(self, key):
        key_first_term = key.partition('.')[0]
        return self.weight_dtype_conversion.get(key_first_term, self.default_dtype)

    def enabled(self):
        return not shared.cmd_opts.disable_model_loading_ram_optimization

    def install(self):
        sd = self.state_dict
        device = self.device

        def load_from_state_dict(original, module, state_dict, prefix, *args, **kwargs):
            used_param_keys = []

            for name, param in module._parameters.items():
                if param is None:
                    continue

                key = prefix + name
                sd_param = sd.pop(key, None)
                if sd_param is not None:
                    state_dict[key] = sd_param.to(dtype=self.get_weight_dtype(key))
                    used_param_keys.append(key)

                if param.is_meta:
                    dtype = sd_param.dtype if sd_param is not None else param.dtype
                    module._parameters[name] = torch.nn.parameter.Parameter(torch.zeros_like(param, device=device, dtype=dtype), requires_grad=param.requires_grad)

            for name in module._buffers:
                key = prefix + name

                sd_param = sd.pop(key, None)
                if sd_param is not None:
                    state_dict[key] = sd_param
                    used_param_keys.append(key)

            result = original(module, state_dict, prefix, *args, **kwargs)

            for key in used_param_keys:
                state_dict.pop(key, None)

            return result

        def load_state_dict(original, module, state_dict, *args, **kwargs):
            """torch makes a lot of copies of the dictionary with weights, so just deleting entries from state_dict does not help
            because the same values are stored in multiple copies of the dict. The trick used here is to give torch a dict with
            all weights on meta device, i.e. deleted, and then it doesn't matter how many copies torch makes.

            In _load_from_state_dict, the correct weight will be obtained from a single dict with the right weights (sd).

            The dangerous thing about this is if _load_from_state_dict is not called, (if some exotic module overloads
            the function and does not call the original) the state dict will just fail to load because weights
            would be on the meta device.
            """

            if state_dict is sd:
                state_dict = {k: v.to(device="meta", dtype=v.dtype) for k, v in state_dict.items()}

            return original(module, state_dict, *args, **kwargs)

        def replace_load_from_state_dict(cls):
            original = self.replace(cls, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(original, *args, **kwargs))

        module_load_state_dict = self.replace(torch.nn.Module, 'load_state_dict', lambda *args, **kwargs: load_state_dict(module_load_state_dict, *args, **kwargs))
        replace_load_from_state_dict(torch.nn.Module)

        # A class with its own hook (the Lora extension installs one on each of these) calls its saved original and so
        # bypasses Module's replacement; it needs its own. The others inherit Module's.
        for cls in (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention, torch.nn.LayerNorm, torch.nn.GroupNorm):
            if '_load_from_state_dict' in vars(cls):
                replace_load_from_state_dict(cls)
