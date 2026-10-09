from scripts.utils import visualize_inpaint_mask
from ..supported_preprocessor import Preprocessor, PreprocessorParameter


class PreprocessorInpaint(Preprocessor):
    """Passes the RGBA input (colour + inpaint mask) through; the ControlNet inpaint model reads the mask channel."""

    def __init__(self, name="inpaint", label="inpaint_global_harmonious", sorting_priority=0):
        super().__init__(name=name)
        self._label = label
        self.tags = ["Inpaint"]
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.sorting_priority = sorting_priority
        self.accepts_mask = True
        self.requires_mask = True

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        return Preprocessor.Result(
            value=input_image,
            display_images=visualize_inpaint_mask(input_image)[None, :, :, :],
        )


class PreprocessorInpaintOnly(PreprocessorInpaint):
    """Same input as inpaint; controlnet.py also pastes the unmasked source pixels back after sampling."""

    def __init__(self):
        super().__init__(name="inpaint_only", label=None, sorting_priority=100)


Preprocessor.add_supported_preprocessor(PreprocessorInpaint())
Preprocessor.add_supported_preprocessor(PreprocessorInpaintOnly())
