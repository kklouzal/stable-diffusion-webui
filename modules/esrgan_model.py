from modules import modelloader, devices
from modules.shared import opts
from modules.upscaler import Upscaler
from modules.upscaler_utils import upscale_with_model


class UpscalerESRGAN(Upscaler):
    name = "ESRGAN"
    model_url = "https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth"
    model_name = "ESRGAN_4x"

    def __init__(self, dirname):
        self.user_path = dirname
        super().__init__()
        self.scalers = self.scalers_from_files([".pt", ".pth"])

    def do_upscale(self, img, selected_model):
        return upscale_with_model(
            self.load_model_or_fail(selected_model),
            img,
            tile_size=opts.ESRGAN_tile,
            tile_overlap=opts.ESRGAN_tile_overlap,
        )

    def load_model(self, path: str):
        # Loaded on the CPU (spandrel's default device), then moved to device_esrgan.
        return modelloader.load_cached_spandrel_model(
            self.local_model_file(path, file_name=f"{self.model_name}.pth"),
            load_device='cpu',
            device=devices.device_esrgan,
            expected_architecture='ESRGAN',
        )
