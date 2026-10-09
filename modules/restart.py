import os
from pathlib import Path

from modules.paths_internal import script_path


def is_restartable() -> bool:
    """
    Return True if the webui is restartable (i.e. there is something watching to restart it with)
    """
    return bool(os.environ.get('SD_WEBUI_RESTART'))


def restart_program() -> None:
    """Creates tmp/restart and exits the process immediately.

    The /server-restart route calls this only when is_restartable(): whatever started the server with
    SD_WEBUI_RESTART set is expected to read tmp/restart as a request to start it again. The GB10 launch scripts
    do not set it.
    """

    tmpdir = Path(script_path) / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    (tmpdir / "restart").touch()

    stop_program()


def stop_program() -> None:
    os._exit(0)
