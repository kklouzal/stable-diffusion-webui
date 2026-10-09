"""Base class and helpers shared by the Incantations submodules (SEG, PAG, CFG combiner) and Dynamic Thresholding."""
from modules import script_callbacks


class UIWrapper:
    """One Incantations submodule. IncantBaseExtensionScript calls each hook with this submodule's slice of the
    script args.

    The script callbacks and forward hooks a submodule installs for a batch are tracked here, so remove_callbacks()
    and remove_hook_handles() release exactly those. Submodules call both at the start of every batch as well as in
    postprocess_batch, because a failed generation skips postprocess_batch.
    """
    def __init__(self):
        self.infotext_fields: list = []
        self._callbacks: list = []
        self._hook_handles: list = []

    def setup_ui(self, is_img2img) -> list:
        raise NotImplementedError

    def get_infotext_fields(self) -> list:
        return self.infotext_fields

    def before_process(self, p, *args, **kwargs):
        pass

    def process(self, p, *args, **kwargs):
        pass

    def before_process_batch(self, p, *args, **kwargs):
        pass

    def process_batch(self, p, *args, **kwargs):
        pass

    def postprocess_batch(self, p, *args, **kwargs):
        pass

    def get_xyz_axis_options(self, xyz_grid) -> list:
        return []

    def track_callback(self, fn):
        """Return ``fn`` after noting it for remove_callbacks(); wrap the ``fn`` passed to a ``script_callbacks.on_*``.

        The submodule's own file must make the ``on_*`` call: A1111 names and reports a callback after the file
        that registered it.
        """
        self._callbacks.append(fn)
        return fn

    def remove_callbacks(self):
        for fn in self._callbacks:
            script_callbacks.remove_callbacks_for_function(fn)
        self._callbacks = []

    def add_forward_hook(self, module, fn):
        """Install ``fn(module, args, kwargs, output)`` as a forward hook of ``module`` until remove_hook_handles()."""
        self._hook_handles.append(module.register_forward_hook(fn, with_kwargs=True))

    def remove_hook_handles(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []


def cond_crossattn(cond):
    """Return the cross-attention tensor of plain or SDXL dict conditioning (None if a dict has none)."""
    return cond.get('crossattn') if isinstance(cond, dict) else cond


def xyz_field_setter(field, active_field, *, boolean=False):
    """Return an X/Y/Z AxisOption apply function for a submodule parameter.

    Stores the axis value on ``p.<field>`` ("true"/"false" become bools when ``boolean``) and defaults
    ``p.<active_field>`` to True if the request left it unset, so plotting a parameter also turns its feature on.
    """
    def apply(p, x, xs):
        if boolean:
            x = x.lower() == "true"
        setattr(p, field, x)
        if not hasattr(p, active_field):
            setattr(p, active_field, True)
    return apply


def add_xyz_axis_options(xyz_grid, axis_options):
    """Append to the X/Y/Z plot script's axis list the options whose label it does not list yet, in order."""
    labels = {option.label for option in xyz_grid.axis_options}
    for option in axis_options:
        if option.label not in labels:
            xyz_grid.axis_options.append(option)
            labels.add(option.label)
