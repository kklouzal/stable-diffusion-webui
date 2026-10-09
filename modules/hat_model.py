import os

from modules import modelloader, devices
from modules.shared import opts
from modules.upscaler import Upscaler
from modules.upscaler_utils import upscale_with_model


class UpscalerHAT(Upscaler):
    name = "HAT"

    def __init__(self, dirname):
        self.user_path = dirname
        super().__init__()
        # TODO: scale might not be 4, but we can't know without loading the model
        self.scalers = self.scalers_from_files([".pt", ".pth"])

    def do_upscale(self, img, selected_model):
        return upscale_with_model(
            self.load_model_or_fail(selected_model),
            img,
            tile_size=opts.ESRGAN_tile,  # TODO: should probably be HAT_tile
            tile_overlap=opts.ESRGAN_tile_overlap,  # TODO: should probably be HAT_tile_overlap
        )

    def load_model(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file {path} not found")
        return modelloader.load_cached_spandrel_model(
            path,
            device=devices.device_esrgan,  # TODO: should probably be device_hat
            expected_architecture='HAT',
        )
