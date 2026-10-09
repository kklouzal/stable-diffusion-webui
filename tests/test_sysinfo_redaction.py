import json
import os
import sys

import pytest

os.environ.setdefault("IGNORE_CMD_ARGS_ERRORS", "1")  # sysinfo imports shared_cmd_options, which parses pytest's argv

from modules import cmd_args, launch_utils, shared, shared_cmd_options, sysinfo  # noqa: E402

# Each command line hands a credential containing "SECRET" to a credential flag (checked against argparse below).
CREDENTIAL_CMDLINES = [
    ["--api-auth", "u:SECRET"],
    ["--api-auth=u:SECRET"],
    ["--listen", "--api-auth", "u1:SECRET,u2:SECRET2", "--port", "7860"],
    ["--gradio-auth", "u:SECRET"],
    ["--gradio-auth=u:SECRET"],
    ["--ngrok", "token:u:SECRET"],
    ['--ngrok-options={"basic_auth": "u:SECRET"}'],
    ["--ngrok-options", '{"authtoken": "SECRET"}'],
    ["--api-a", "u:SECRET"],  # argparse abbreviation of --api-auth
    ["--api-a=u:SECRET"],
    ["--ngrok-o=" + '{"basic_auth": "u:SECRET"}'],
]


def test_credential_flag_dests_are_parser_flags():
    parsed, _ = cmd_args.parser.parse_known_args([])
    assert launch_utils.credential_flag_dests <= set(vars(parsed))


@pytest.mark.parametrize("tokens", CREDENTIAL_CMDLINES)
def test_redact_cmdline_hides_every_credential_argparse_accepts(tokens):
    parsed, unknown = cmd_args.parser.parse_known_args(tokens)
    assert not unknown
    assert any("SECRET" in str(getattr(parsed, dest)) for dest in launch_utils.credential_flag_dests)

    redacted = launch_utils.redact_cmdline(tokens)

    assert len(redacted) == len(tokens)
    assert not any("SECRET" in token for token in redacted)


@pytest.mark.parametrize(("tokens", "expected"), [
    (["--api-auth", "u:p", "--listen"], ["--api-auth", "<hidden>", "--listen"]),
    (["--api-auth=u:p", "--listen"], ["--api-auth=<hidden>", "--listen"]),
    (['--ngrok-options={"a": "b=c"}'], ["--ngrok-options=<hidden>"]),
    # exact non-credential flags that a credential flag name starts with, or that start with one
    (["--api", "--gradio-auth-path", "/auth.txt", "--port", "7860"], ["--api", "--gradio-auth-path", "/auth.txt", "--port", "7860"]),
    (["--styles-file", "a=b.csv", "--listen"], ["--styles-file", "a=b.csv", "--listen"]),
])
def test_redact_cmdline_keeps_everything_else(tokens, expected):
    assert launch_utils.redact_cmdline(tokens) == expected


def test_dump_sysinfo_hides_credentials_in_argv_and_commandline_args(monkeypatch, tmp_path):
    # launch.py --dump-sysinfo runs before initialization: no shared.opts, so the dump reads the settings file
    settings_file = tmp_path / "config.json"
    settings_file.write_text(json.dumps({"disabled_extensions": []}), encoding="utf8")
    monkeypatch.setattr(shared, "opts", None)
    monkeypatch.setattr(shared_cmd_options.cmd_opts, "ui_settings_file", str(settings_file))
    monkeypatch.setattr(sys, "argv",["launch.py", "--dump-sysinfo", "--api-auth=u1:SECRET_A", "--gradio-auth", "u2:SECRET_B", "--listen"])
    monkeypatch.setenv("COMMANDLINE_ARGS", """--listen --api-auth u3:SECRET_C '--ngrok-options={"basic_auth": "u4:SECRET_D"}'""")
    monkeypatch.chdir(tmp_path)

    filename = launch_utils.dump_sysinfo()

    text = (tmp_path / filename).read_text(encoding="utf8")
    assert "SECRET" not in text
    dump = json.loads(text)
    assert dump["Commandline"] == ["launch.py", "--dump-sysinfo", "--api-auth=<hidden>", "--gradio-auth", "<hidden>", "--listen"]
    assert dump["Environment"]["COMMANDLINE_ARGS"] == "--listen --api-auth '<hidden>' '--ngrok-options=<hidden>'"
    assert set(dump["Environment"]) <= sysinfo.environment_whitelist
    assert dump["Config"] == {"disabled_extensions": []}
