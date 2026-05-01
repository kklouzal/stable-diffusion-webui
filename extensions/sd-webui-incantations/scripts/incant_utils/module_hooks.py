from typing import Optional
import logging
import torch


from modules import shared


logger = logging.getLogger(__name__)


def modules_add_field(modules, field, value=None):
    """ Add a field to a module if it isn't already added.
    Args:
        modules (list): Module or list of modules to add the field to
        field (str): Field name to add
        value (any): Value to assign to the field
    Returns:
        None

    """
    if not isinstance(modules, list):
        modules = [modules]
    for module in modules:
        if not hasattr(module, field):
            setattr(module, field, value)
        else:
            logger.warning(f"Field {field} already exists in module {module}")


def modules_remove_field(modules, field):
    """ Remove a field from a module if it exists.
    Args:
        modules (list): Module or list of modules to add the field to
        field (str): Field name to add
        value (any): Value to assign to the field
    Returns:
        None

    """
    if not isinstance(modules, list):
        modules = [modules]
    for module in modules:
        if hasattr(module, field):
                delattr(module, field)
        else:
            # Field already absent; cleanup is idempotent.
            continue


def get_modules(network_layer_name_filter: Optional[str] = None, module_name_filter: Optional[str] = None):
    """ Get all modules from the shared.sd_model that match the filters provided. If no filters are provided, all modules are returned.

    Args:
        network_layer_name_filter (Optional[str], optional): Filters the modules by network layer name. Defaults to None. Example: "attn1" will return all modules that have "attn1" in their network layer name.
        module_name_filter (Optional[str], optional): Filters the modules by module class name. Defaults to None. Example: "CrossAttention" will return all modules that have "CrossAttention" in their class name.

    Returns:
        list: List of modules that match the filters provided.
    """
    try:
        m = shared.sd_model
        nlm = m.network_layer_mapping
        sd_model_modules = nlm.values()

        # Apply filters if they are provided
        if network_layer_name_filter is not None:
            sd_model_modules = list(filter(lambda m: network_layer_name_filter in m.network_layer_name, sd_model_modules))
        if module_name_filter is not None:
            sd_model_modules = list(filter(lambda m: module_name_filter in m.__class__.__name__, sd_model_modules))
        return sd_model_modules
    except AttributeError:
        logger.exception("AttributeError in get_modules", stack_info=True)
        return []
    except Exception:
        logger.exception("Exception in get_modules", stack_info=True)
        return []



def module_add_forward_hook(module, hook_fn, hook_type="forward", with_kwargs=False):
    """ Adds a forward hook to a module.

    hook_fn should be a function that accepts the following arguments:
        forward hook, no kwargs: hook(module, args, output) -> None or modified output
        forward hook, with kwargs: hook(module, args, kwargs output) -> None or modified output

    Args:
        module (torch.nn.Module): Module to hook
        hook_fn (Callable): Function to call when the hook is triggered
        hook_type (str, optional): Type of hook to create. Defaults to "forward". Can be "forward" or "pre_forward".
        with_kwargs (bool, optional): Whether the hook function should accept keyword arguments. Defaults to False.

    Returns:
        torch.utils.hooks.RemovableHandle: Handle for the hook
    """
    if module is None:
        raise ValueError("module must be provided")
    if not callable(hook_fn):
        raise ValueError("hook_fn must be a callable function")

    if hook_type == "forward":
        handle = module.register_forward_hook(hook_fn, with_kwargs=with_kwargs)
    elif hook_type == "pre_forward":
        handle = module.register_forward_pre_hook(hook_fn, with_kwargs=with_kwargs)
    else:
        raise ValueError(f"Invalid hook type {hook_type}. Must be 'forward' or 'pre_forward'.")

    return handle