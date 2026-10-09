from ..ipadapter.presets import IPAdapterPreset
from ..supported_preprocessor import Preprocessor
from ..logging import logger


class PreprocessorIPAdapterAuto(Preprocessor):
    def __init__(self):
        super().__init__(name="ip-adapter-auto")
        self.tags = ["IP-Adapter"]
        self.sorting_priority = 1000
        self.returns_image = False

    @staticmethod
    def get_preprocessor_by_model(model):
        module: str = IPAdapterPreset.match_model(model).module
        return Preprocessor.get_preprocessor(module)

    def __call__(self, *args, **kwargs):
        assert "model" in kwargs
        model: str = kwargs["model"]
        p = PreprocessorIPAdapterAuto.get_preprocessor_by_model(model)
        assert p is not None, f"ip-adapter-auto: no registered preprocessor for model {model}"
        logger.info(f"ip-adapter-auto => {p.label}")
        return p(*args, **kwargs)


Preprocessor.add_supported_preprocessor(PreprocessorIPAdapterAuto())
