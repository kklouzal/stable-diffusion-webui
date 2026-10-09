"""Importing this package registers every preprocessor (Preprocessor.add_supported_preprocessor)."""
from . import teed  # noqa: F401
from . import inpaint  # noqa: F401
from . import lama_inpaint  # noqa: F401
from . import ip_adapter_auto  # noqa: F401
from . import model_free_preprocessors  # noqa: F401
from .legacy import legacy_preprocessors  # noqa: F401
from . import mobile_sam  # noqa: F401
