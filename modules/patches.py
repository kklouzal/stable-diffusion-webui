from collections import defaultdict


def patch(key, obj, field, replacement):
    """Replaces a function in a module or a class.

    Also stores the original function in this module, possible to be retrieved via original(key, obj, field).
    If the function is already replaced by this caller (key), an exception is raised -- use undo() before that.

    Arguments:
        key: identifying information for who is doing the replacement. You can use __name__.
        obj: the module or the class
        field: name of the function as a string
        replacement: the new function

    Returns:
        the original function
    """

    patch_key = (obj, field)
    if patch_key in originals[key]:
        raise RuntimeError(f"patch for {field} is already applied")

    original_func = getattr(obj, field)
    originals[key][patch_key] = original_func

    setattr(obj, field, replacement)

    return original_func



originals = defaultdict(dict)
