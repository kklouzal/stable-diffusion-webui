import PIL.Image

import modules.upscaler
from modules import devices, modelloader, script_callbacks, shared, upscaler_utils


class UpscalerScuNET(modules.upscaler.Upscaler):
    name = "ScuNET"
    model_name = "ScuNET GAN"
    model_url = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth"
    model_name2 = "ScuNET PSNR"
    model_url2 = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"

    def __init__(self, dirname):
        self.user_path = dirname
        super().__init__()
        self.scalers = self.scalers_from_files([".pth"])
        if not any(scaler.name == self.model_name2 or scaler.data_path == self.model_url2 for scaler in self.scalers):
            self.scalers.append(modules.upscaler.UpscalerData(self.model_name2, self.model_url2, self))

    def do_upscale(self, img: PIL.Image.Image, selected_file):
        devices.torch_gc()
        model = self.load_model_or_fail(selected_file)

        img = upscaler_utils.upscale_2(
            img,
            model,
            tile_size=shared.opts.SCUNET_tile,
            tile_overlap=shared.opts.SCUNET_tile_overlap,
            scale=1,  # ScuNET is a denoising model, not an upscaler
            desc='ScuNET',
        )
        devices.torch_gc()
        return img

    def load_model(self, path: str):
        # A URL model is saved under the URL's basename: spandrel picks its reader from the .pth extension.
        return modelloader.load_cached_spandrel_model(
            self.local_model_file(path),
            device=devices.get_device_for('scunet'),
            expected_architecture='SCUNet',
        )


def on_ui_settings():
    from modules import headless_ui as gr

    shared.opts.add_option("SCUNET_tile", shared.OptionInfo(256, "Tile size for SCUNET upscalers.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}, section=('upscaling', "Upscaling")).info("0 = no tiling"))
    shared.opts.add_option("SCUNET_tile_overlap", shared.OptionInfo(8, "Tile overlap for SCUNET upscalers.", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}, section=('upscaling', "Upscaling")).info("Low values = visible seam"))


script_callbacks.on_ui_settings(on_ui_settings)
