class UIWrapper:
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = []

    def setup_ui(self, is_img2img) -> list:
        raise NotImplementedError

    def get_infotext_fields(self) -> list:
        return self.infotext_fields

    def get_paste_field_names(self) -> list:
        return self.paste_field_names

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

    def get_xyz_axis_options(self) -> list:
        return []


def xyz_field_setter(field, active_field, *, boolean=False, also_enable=None):
    """Return an X/Y/Z AxisOption apply function for a submodule parameter.

    Stores the axis value on ``p.<field>`` ("true"/"false" become bools when
    ``boolean``) and defaults ``p.<active_field>`` -- and ``p.<also_enable>`` when
    given -- to True if the request left it unset, so plotting a parameter also
    turns its feature on.
    """
    def apply(p, x, xs):
        if boolean:
            x = x.lower() == "true"
        setattr(p, field, x)
        for flag in (active_field, also_enable):
            if flag is not None and not hasattr(p, flag):
                setattr(p, flag, True)
    return apply
