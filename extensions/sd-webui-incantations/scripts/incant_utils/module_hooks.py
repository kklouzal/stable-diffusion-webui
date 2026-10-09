from typing import Optional
import logging


from modules import shared


logger = logging.getLogger(__name__)


def modules_add_field(module, field, value=None):
    """ Add a field to a module if it isn't already added.
    Args:
        module: Module to add the field to
        field (str): Field name to add
        value (any): Value to assign to the field
    Returns:
        None

    """
    if not hasattr(module, field):
        setattr(module, field, value)
    else:
        logger.warning(f"Field {field} already exists in module {module}")


def modules_remove_field(module, field):
    """ Remove a field from a module if it exists (cleanup is idempotent).
    Args:
        module: Module to remove the field from
        field (str): Field name to remove
    Returns:
        None

    """
    if hasattr(module, field):
        delattr(module, field)


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
