# launch.py's runtime half: launch arguments, source version info, the extension list shared_cmd_options needs before
# its parser exists, the sysinfo dump and the server start.
import os
import subprocess
import sys
import json
import shlex
from functools import lru_cache

from modules import cmd_args, errors
from modules.paths_internal import script_path, extensions_dir
from modules.timer import startup_timer
from modules import logging_config

args, _ = cmd_args.parser.parse_known_args()
logging_config.setup_logging(args.loglevel)

git = os.environ.get('GIT', "git")


def _webui_source_version_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


@lru_cache()
def _script_path_is_git_repo():
    return os.path.isdir(os.path.join(script_path, ".git"))


@lru_cache()
def commit_hash():
    env_commit = _webui_source_version_env("A1111_COMMIT_HASH")
    if env_commit is not None:
        return env_commit
    if not _script_path_is_git_repo():
        return "<none>"
    try:
        return subprocess.check_output([git, "-C", script_path, "rev-parse", "HEAD"], shell=False, stderr=subprocess.DEVNULL, encoding='utf8').strip()
    except Exception:
        return "<none>"


@lru_cache()
def git_tag():
    env_tag = _webui_source_version_env("A1111_VERSION_TAG")
    if env_tag is not None:
        return env_tag
    if _script_path_is_git_repo():
        try:
            return subprocess.check_output([git, "-C", script_path, "describe", "--tags"], shell=False, stderr=subprocess.DEVNULL, encoding='utf8').strip()
        except Exception:
            pass
    try:
        changelog_md = os.path.join(script_path, "CHANGELOG.md")
        with open(changelog_md, "r", encoding="utf-8") as file:
            line = next((line.strip() for line in file if line.strip()), "<none>")
            line = line.replace("## ", "")
            return line
    except Exception:
        return "<none>"


def list_extensions(settings_file):
    settings = {}

    try:
        with open(settings_file, "r", encoding="utf8") as file:
            settings = json.load(file)
    except FileNotFoundError:
        pass
    except Exception:
        errors.report(f'\nCould not load settings\nThe config file "{settings_file}" is likely corrupted\nIt has been moved to the "tmp/config.json"\nReverting config to default\n\n''', exc_info=True)
        os.replace(settings_file, os.path.join(script_path, "tmp", "config.json"))

    disabled_extensions = set(settings.get('disabled_extensions', []))
    disable_all_extensions = settings.get('disable_all_extensions', 'none')

    if disable_all_extensions != 'none' or args.disable_extra_extensions or args.disable_all_extensions or not os.path.isdir(extensions_dir):
        return []

    return [x for x in os.listdir(extensions_dir) if x not in disabled_extensions]


def start():
    startup_timer.record("initial startup")
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {shlex.join(sys.argv[1:])}")
    import webui
    if '--nowebui' in sys.argv:
        webui.api_only()
    else:
        webui.webui()


def dump_sysinfo():
    from modules import sysinfo
    import datetime

    text = sysinfo.get()
    filename = f"sysinfo-{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d-%H-%M')}.json"

    with open(filename, "w", encoding="utf8") as file:
        file.write(text)

    return filename
