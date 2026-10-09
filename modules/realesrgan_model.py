from modules import modelloader
from modules.shared import cmd_opts, opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model


class UpscalerRealESRGAN(Upscaler):
    name = "RealESRGAN"

    def __init__(self, path):
        self.user_path = path
        super().__init__()
        self.scalers = []
        scalers = get_realesrgan_models(self)

        local_model_paths = self.find_models(ext_filter=[".pth"])
        for scaler in scalers:
            if scaler.local_data_path.startswith("http"):
                filename = modelloader.friendly_name(scaler.local_data_path)
                local_model_candidates = [local_model for local_model in local_model_paths if local_model.endswith(f"{filename}.pth")]
                if local_model_candidates:
                    scaler.local_data_path = local_model_candidates[0]

            if scaler.name in opts.realesrgan_enabled_models:
                self.scalers.append(scaler)

    def do_upscale(self, img, path):
        return upscale_with_model(
            self.load_model_or_fail(path),
            img,
            tile_size=opts.ESRGAN_tile,
            tile_overlap=opts.ESRGAN_tile_overlap,
            # TODO: `outscale`?
        )

    def load_model(self, path):
        return modelloader.load_cached_spandrel_model(
            self.listed_model_file(path),
            device=self.device,
            prefer_half=(not cmd_opts.no_half and not cmd_opts.upcast_sampling),
            expected_architecture="ESRGAN",  # "RealESRGAN" isn't a specific thing for Spandrel
        )


def get_realesrgan_models(scaler: UpscalerRealESRGAN):
    return [
        UpscalerData(
            name="R-ESRGAN General 4xV3",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN General WDN 4xV3",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN AnimeVideo",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN 4x+",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN 4x+ Anime6B",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN 2x+",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            scale=2,
            upscaler=scaler,
        ),
    ]
