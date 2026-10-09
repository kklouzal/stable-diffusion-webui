from modules import modelloader
from modules.shared import cmd_opts, opts, hf_endpoint
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model


class UpscalerDAT(Upscaler):
    name = "DAT"

    def __init__(self, user_path):
        self.user_path = user_path
        super().__init__()
        self.scalers = self.scalers_from_files([".pt", ".pth"], scale=None)
        self.scalers += [model for model in get_dat_models(self) if model.name in opts.dat_enabled_models]

    def do_upscale(self, img, path):
        return upscale_with_model(
            self.load_model_or_fail(path),
            img,
            tile_size=opts.DAT_tile,
            tile_overlap=opts.DAT_tile_overlap,
        )

    def load_model(self, path):
        return modelloader.load_cached_spandrel_model(
            # 200 bytes: a cached file that small is a Git LFS pointer, not the weights.
            self.listed_model_file(path, redownload_below_bytes=200),
            device=self.device,
            prefer_half=(not cmd_opts.no_half and not cmd_opts.upcast_sampling),
            expected_architecture="DAT",
        )


def get_dat_models(scaler):
    return [
        UpscalerData(
            name="DAT x2",
            path=f"{hf_endpoint}/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x2.pth",
            scale=2,
            upscaler=scaler,
            sha256='7760aa96e4ee77e29d4f89c3a4486200042e019461fdb8aa286f49aa00b89b51',
        ),
        UpscalerData(
            name="DAT x3",
            path=f"{hf_endpoint}/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x3.pth",
            scale=3,
            upscaler=scaler,
            sha256='581973e02c06f90d4eb90acf743ec9604f56f3c2c6f9e1e2c2b38ded1f80d197',
        ),
        UpscalerData(
            name="DAT x4",
            path=f"{hf_endpoint}/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x4.pth",
            scale=4,
            upscaler=scaler,
            sha256='391a6ce69899dff5ea3214557e9d585608254579217169faf3d4c353caff049e',
        ),
    ]
