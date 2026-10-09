import math
import os
from abc import abstractmethod

import PIL
from PIL import Image

from modules import modelloader, shared


def scaled_size(size: int, scale: float) -> int:
    """`int(size * scale)`, except that a product a few ULPs below an integer counts as that integer.

    Callers derive `scale` from a target size (`target / size`), and the float product `size * (target / size)`
    can land just below `target` (616 * (640 / 616) == 639.9999999999999); truncating that would make the
    upscaler miss the requested size by a pixel, or by 8 once floored to a multiple of 8.
    """
    return math.floor(size * scale + 1e-6)


class Upscaler:
    """Base of every upscaler. `modelloader.load_upscalers` instantiates each direct subclass with its
    `--<classname minus "upscaler">-models-path` flag value, so shared behaviour lives here as methods rather than in
    an intermediate class."""

    name = None
    model_path = None
    model_name = None
    model_url = None
    user_path = None
    scalers: list

    def __init__(self, create_dirs=False):
        self.device = shared.device
        self.scale = 1
        self.model_download_path = None

        if self.model_path is None and self.name:
            self.model_path = os.path.join(shared.models_path, self.name)
        if self.model_path and create_dirs:
            os.makedirs(self.model_path, exist_ok=True)

    @abstractmethod
    def do_upscale(self, img: PIL.Image, selected_model: str):
        return img

    def upscale(self, img: PIL.Image, scale, selected_model: str = None):
        self.scale = scale
        dest_w = scaled_size(img.width, scale) // 8 * 8
        dest_h = scaled_size(img.height, scale) // 8 * 8

        for i in range(3):
            if img.width >= dest_w and img.height >= dest_h and (i > 0 or scale != 1):
                break

            if shared.state.interrupted:
                break

            shape = (img.width, img.height)

            img = self.do_upscale(img, selected_model)

            if shape == (img.width, img.height):
                break

        if img.width != dest_w or img.height != dest_h:
            img = img.resize((int(dest_w), int(dest_h)), resample=Image.Resampling.LANCZOS)

        return img

    @abstractmethod
    def load_model(self, path: str):
        pass

    def load_model_or_fail(self, path: str):
        """`load_model(path)`, raising RuntimeError on any failure: returning the input image instead would silently
        LANCZOS-resize it while infotext still names this model."""
        try:
            return self.load_model(path)
        except Exception as e:
            raise RuntimeError(f"Unable to load {self.name} model {path}: {e}") from e

    def find_models(self, ext_filter=None) -> list:
        return modelloader.load_models(model_path=self.model_path, model_url=self.model_url, command_path=self.user_path, ext_filter=ext_filter)

    def scalers_from_files(self, ext_filter, scale=4) -> list:
        """One `UpscalerData` per model file found; with none found, `find_models` lists `model_url` instead, which is
        named `model_name`."""
        return [
            UpscalerData(self.model_name if path.startswith("http") else modelloader.friendly_name(path), path, self, scale)
            for path in self.find_models(ext_filter=ext_filter)
        ]

    def local_model_file(self, path: str, file_name: str = None) -> str:
        """`path`, or for a URL the file downloaded from it into `model_download_path` (as `file_name` if given, else
        under the URL's basename), downloading only when that file is missing."""
        if path.startswith("http"):
            return modelloader.load_file_from_url(path, model_dir=self.model_download_path, file_name=file_name)
        return path

    def listed_model_file(self, path: str, redownload_below_bytes: int = None) -> str:
        """The local file of the listed scaler whose `data_path` is `path`. A URL is downloaded on first use, checked
        against the scaler's `sha256` if it has one, and downloaded again if the file is smaller than
        `redownload_below_bytes` (a Git LFS pointer instead of the weights)."""
        for scaler in self.scalers:
            if scaler.data_path == path:
                if scaler.local_data_path.startswith("http"):
                    scaler.local_data_path = modelloader.load_file_from_url(
                        scaler.data_path,
                        model_dir=self.model_download_path,
                        hash_prefix=scaler.sha256,
                    )

                    if redownload_below_bytes is not None and os.path.getsize(scaler.local_data_path) < redownload_below_bytes:
                        scaler.local_data_path = modelloader.load_file_from_url(
                            scaler.data_path,
                            model_dir=self.model_download_path,
                            hash_prefix=scaler.sha256,
                            re_download=True,
                        )

                if not os.path.exists(scaler.local_data_path):
                    raise FileNotFoundError(f"{self.name} data missing: {scaler.local_data_path}")
                return scaler.local_data_path
        raise ValueError(f"Unable to find model info: {path}")


class UpscalerData:
    name = None
    data_path = None
    scale: int = 4
    scaler: Upscaler = None
    model: None

    def __init__(self, name: str, path: str, upscaler: Upscaler = None, scale: int = 4, model=None, sha256: str = None):
        self.name = name
        self.data_path = path
        self.local_data_path = path
        self.scaler = upscaler
        self.scale = scale
        self.model = model
        self.sha256 = sha256

    def __repr__(self):
        return f"<UpscalerData name={self.name} path={self.data_path} scale={self.scale}>"


class UpscalerNone(Upscaler):
    name = "None"
    scalers = []

    def load_model(self, path):
        pass

    def do_upscale(self, img, selected_model=None):
        return img

    def __init__(self, dirname=None):
        super().__init__(False)
        self.scalers = [UpscalerData("None", None, self)]


class UpscalerLanczos(Upscaler):
    scalers = []

    def do_upscale(self, img, selected_model=None):
        return img.resize((scaled_size(img.width, self.scale), scaled_size(img.height, self.scale)), resample=Image.Resampling.LANCZOS)

    def load_model(self, _):
        pass

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Lanczos"
        self.scalers = [UpscalerData("Lanczos", None, self)]


class UpscalerNearest(Upscaler):
    scalers = []

    def do_upscale(self, img, selected_model=None):
        return img.resize((scaled_size(img.width, self.scale), scaled_size(img.height, self.scale)), resample=Image.Resampling.NEAREST)

    def load_model(self, _):
        pass

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Nearest"
        self.scalers = [UpscalerData("Nearest", None, self)]
