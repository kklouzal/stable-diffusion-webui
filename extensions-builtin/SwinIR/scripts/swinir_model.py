import torch
from PIL import Image

from modules import devices, modelloader, script_callbacks, shared, upscaler_utils
from modules.upscaler import Upscaler


class UpscalerSwinIR(Upscaler):
    name = "SwinIR"
    model_url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
    model_name = "SwinIR 4x"

    def __init__(self, dirname):
        self.user_path = dirname
        super().__init__()
        self.scalers = self.scalers_from_files([".pt", ".pth"])

    def do_upscale(self, img: Image.Image, model_file: str) -> Image.Image:
        model = self.load_model_or_fail(model_file)
        return upscaler_utils.upscale_2(
            img,
            model,
            tile_size=shared.opts.SWIN_tile,
            tile_overlap=shared.opts.SWIN_tile_overlap,
            scale=model.scale,
            desc="SwinIR",
        )

    def load_model(self, path):
        # The shared cache also keeps a compiled model, so SWIN_torch_compile compiles once per model file and device.
        return modelloader.load_cached_spandrel_model(
            self.local_model_file(path, file_name=f"{self.model_name.replace(' ', '_')}.pth"),
            device=self._get_device(),
            prefer_half=(devices.dtype == torch.float16),
            expected_architecture="SwinIR",
            compile_model=bool(getattr(shared.opts, 'SWIN_torch_compile', False)),
        )

    def _get_device(self):
        return devices.get_device_for('swinir')


def on_ui_settings():
    from modules import headless_ui as gr

    shared.opts.add_option("SWIN_tile", shared.OptionInfo(192, "Tile size for all SwinIR.", gr.Slider, {"minimum": 16, "maximum": 512, "step": 16}, section=('upscaling', "Upscaling")))
    shared.opts.add_option("SWIN_tile_overlap", shared.OptionInfo(8, "Tile overlap, in pixels for SwinIR. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}, section=('upscaling', "Upscaling")))
    shared.opts.add_option("SWIN_torch_compile", shared.OptionInfo(False, "Use torch.compile to accelerate SwinIR.", gr.Checkbox, {"interactive": True}, section=('upscaling', "Upscaling")).info("Takes longer on first run"))


script_callbacks.on_ui_settings(on_ui_settings)
