"""Live-server API tests: they send real requests (generations, /sdapi/v1/options writes) to a running webui.

They are skipped unless a server is named explicitly through pytest-base-url:

    pytest test/live --base-url http://127.0.0.1:7860

Never point them at a production server: they queue GPU work and rewrite its saved settings.
"""
import base64
from pathlib import Path

import pytest

from test.helpers import TEST_FILES

LIVE_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    if config.getoption("base_url", default=None):
        return
    skip = pytest.mark.skip(reason="live server test: pass --base-url URL (pytest-base-url) to run it")
    for item in items:
        if item.path.is_relative_to(LIVE_DIR):
            item.add_marker(skip)


def _file_to_base64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


@pytest.fixture(scope="session")
def img2img_basic_image_base64() -> str:
    return _file_to_base64(TEST_FILES / "img2img_basic.png")


@pytest.fixture(scope="session")
def mask_basic_image_base64() -> str:
    return _file_to_base64(TEST_FILES / "mask_basic.png")
