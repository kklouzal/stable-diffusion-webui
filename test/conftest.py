import os

import pytest


def pytest_configure(config):
    # We don't want to fail on Py.test command line arguments being
    # parsed by webui:
    os.environ.setdefault("IGNORE_CMD_ARGS_ERRORS", "1")


@pytest.fixture(scope="session")
def initialize() -> None:
    from modules import shared

    # webui's import runs initialize.imports(), whose shared_init.initialize() builds a new shared.opts (and state,
    # ...). When an earlier test module already initialized shared, modules imported since keep the objects they
    # bound with `from modules.shared import opts`, so options later tests set on shared.opts would not reach them.
    if getattr(shared, "opts", None) is None:
        import webui  # noqa: F401
