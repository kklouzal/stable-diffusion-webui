# launch.py's runtime half: launch arguments, source version info, the extension list shared_cmd_options needs before
# its parser exists, command-line redaction, the sysinfo dump and the server start.
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
    # The image has no .git: gb10/run.sh passes the host checkout's `git describe --tags` as A1111_VERSION_TAG.
    env_tag = _webui_source_version_env("A1111_VERSION_TAG")
    if env_tag is not None:
        return env_tag
    if not _script_path_is_git_repo():
        return "<none>"
    try:
        return subprocess.check_output([git, "-C", script_path, "describe", "--tags"], shell=False, stderr=subprocess.DEVNULL, encoding='utf8').strip()
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


# Destinations of the flags whose values are credentials: "user:password" lists and the ngrok authtoken and options.
credential_flag_dests = frozenset({"api_auth", "gradio_auth", "ngrok", "ngrok_options"})


def _is_credential_flag(flag):
    actions = cmd_args.parser._option_string_actions
    if flag in actions:
        return actions[flag].dest in credential_flag_dests

    # argparse also accepts any unambiguous prefix of a '--' flag
    return flag.startswith("--") and len(flag) > 2 and any(option.startswith(flag) and action.dest in credential_flag_dests for option, action in actions.items())


def redact_cmdline(tokens):
    """Returns a copy of the command-line tokens with the value of every credential flag replaced by "<hidden>".

    Covers the '--flag value' and '--flag=value' forms and argparse's '--fl' abbreviations. On a command line that
    argparse would reject it may hide more than a value, never less.
    """
    res = []
    hide_value = False
    for token in tokens:
        if hide_value:
            res.append("<hidden>")
            hide_value = False
            continue

        flag, has_value, _ = token.partition("=")
        if _is_credential_flag(flag):
            if has_value:
                token = f"{flag}=<hidden>"
            else:
                hide_value = True

        res.append(token)

    return res


def start():
    startup_timer.record("initial startup")
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {shlex.join(redact_cmdline(sys.argv[1:]))}")
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
