"""gb10/run.sh, run for real against stand-in docker/sudo/rsync/curl and a stand-in smoke test: the replaced container
is kept until the new one answers the API and passes the smoke test, and any failure after the stop puts the replaced
container (and the owned extension sources) back and exits non-zero. The image runs by ID, with the version its
provenance labels record."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import signal
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NEW_ID = "sha256:" + "1" * 64
OLD_ID = "sha256:" + "0" * 64
UNLABELED_ID = "sha256:" + "2" * 64
NEW_LABELS = {
    "org.opencontainers.image.revision": "c0ffee",
    "org.opencontainers.image.version": "quality-locked-baseline-20260812T004453Z-777-gc0ffee",
}

FAKE_DOCKER = r'''
import json, os, sys

STATE = os.environ["FAKE_DOCKER_STATE"]
with open(STATE) as f:
    state = json.load(f)
argv = sys.argv[1:]
state["calls"].append(argv)


def save():
    with open(STATE, "w") as f:
        json.dump(state, f)


def fail(message):
    save()
    print(message, file=sys.stderr)
    sys.exit(1)


def option(name):
    return argv[argv.index(name) + 1] if name in argv else None


def positional(args):
    out, skip = [], False
    for arg in args:
        if skip:
            skip = False
        elif arg in ("--format", "--filter", "-t", "--tail"):
            skip = True
        elif not arg.startswith("-"):
            out.append(arg)
    return out


def resolve_image(ref):
    image_id = state["tags"].get(ref, ref)
    if image_id not in state["images"]:
        fail(f"Error: No such image: {ref}")
    return image_id


containers = state["containers"]
command = argv[0]
if command == "image" and argv[1] == "inspect":
    image_id = resolve_image(positional(argv[2:])[0])
    fmt = option("--format")
    if fmt == "{{.Id}}":
        print(image_id)
    elif fmt.startswith("{{index .Config.Labels "):
        print(state["images"][image_id].get(fmt.split('"')[1], ""))
    else:
        fail(f"unsupported image format {fmt}")
elif command == "ps":
    name = option("--filter").removeprefix("name=^/").removesuffix("$")
    if name in containers:
        print(name)
elif command == "inspect":
    name = positional(argv[1:])[0]
    if name not in containers:
        fail(f"Error: No such object: {name}")
    c = containers[name]
    fmt = option("--format")
    values = {
        "{{.RestartCount}}": str(c["restarts"]),
        "{{.State.Status}} {{.RestartCount}}": f'{c["status"]} {c["restarts"]}',
        "{{.State.Status}}": c["status"],
        "{{.Image}}": c["image"],
        "{{range .Config.Env}}{{println .}}{{end}}": "".join(e + "\n" for e in c["env"]),
    }
    print(values[fmt], end="" if fmt.startswith("{{range") else "\n")
elif command == "stop":
    name = positional(argv[1:])[0]
    containers[name]["status"] = "exited"
elif command == "rm":
    name = positional(argv[1:])[0]
    if containers[name]["status"] == "running":
        fail("Error: cannot remove a running container")
    del containers[name]
elif command == "rename":
    old, new = positional(argv[1:])
    if new in containers:
        fail(f"Error: name {new} in use")
    containers[new] = containers.pop(old)
elif command == "start":
    name = positional(argv[1:])[0]
    c = containers[name]
    c["status"] = "exited" if c["image"] in state["crash_images"] else "running"
elif command == "run":
    image_id = resolve_image(argv[-1])
    name = option("--name")
    if name in containers:
        fail(f"Error: name {name} in use")
    env = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-e"]
    containers[name] = {"image": image_id, "status": "exited" if image_id in state["crash_images"] else "running", "restarts": 0, "env": env, "run_args": argv}
elif command == "logs":
    print("fake container log line")
else:
    fail(f"unsupported fake docker command {argv}")
save()
'''

FAKE_SUDO = r'''#!/bin/sh
# Runs the command as the current user; the ownership operations the deploy does as root become checks that the
# arguments are well formed. Every command is recorded in $FAKE_SUDO_LOG.
printf '%s\n' "$*" >> "$FAKE_SUDO_LOG"
case "$1" in
  chown) exit 0 ;;
  install) shift; args=""; while [ $# -gt 0 ]; do case "$1" in -o|-g) shift 2 ;; *) args="$args $1"; shift ;; esac; done; exec install $args ;;
  setpriv) while [ "$1" != "test" ]; do shift; done; exec "$@" ;;
esac
exec "$@"
'''

# rsync stand-in: copies SRC/ into DST/ honouring --delete, 'P /dir/***' protection and the --exclude forms run.sh
# uses ('/dir/' anchored, 'dir/' anywhere, '*.pyc' by name).
FAKE_RSYNC = r'''
import fnmatch, os, shutil, sys

args = sys.argv[1:]
src, dst = args[-2], args[-1]
delete = "--delete" in args
excludes = [args[i + 1] for i, a in enumerate(args) if a == "--exclude"]
protected = [args[i + 1][2:].rstrip("*").strip("/") for i, a in enumerate(args) if a == "--filter" and args[i + 1].startswith("P ")]
if not src.endswith("/"):
    dst = os.path.join(dst, os.path.basename(src.rstrip("/")))
    src = src + "/"


def excluded(rel, is_dir):
    name = os.path.basename(rel)
    for pattern in excludes:
        if pattern.endswith("/"):
            if is_dir and (pattern.startswith("/") and rel == pattern.strip("/") or not pattern.startswith("/") and name == pattern.strip("/")):
                return True
        elif fnmatch.fnmatch(name, pattern):
            return True
    return False


wanted = set()
for root, dirs, files in os.walk(src):
    rel_root = os.path.relpath(root, src)
    rel_root = "" if rel_root == "." else rel_root
    dirs[:] = [d for d in dirs if not excluded(os.path.join(rel_root, d), True)]
    os.makedirs(os.path.join(dst, rel_root), exist_ok=True)
    wanted.add(rel_root)
    for d in dirs:
        wanted.add(os.path.join(rel_root, d))
    for name in files:
        rel = os.path.join(rel_root, name)
        if not excluded(rel, False):
            wanted.add(rel)
            shutil.copy2(os.path.join(src, rel), os.path.join(dst, rel))
if delete:
    for root, dirs, files in os.walk(dst, topdown=False):
        rel_root = os.path.relpath(root, dst)
        rel_root = "" if rel_root == "." else rel_root
        for name in files + dirs:
            rel = os.path.join(rel_root, name)
            if rel not in wanted and not any(rel == p or rel.startswith(p + "/") for p in protected):
                path = os.path.join(dst, rel)
                shutil.rmtree(path) if os.path.isdir(path) and not os.path.islink(path) else os.remove(path)
'''

# While the file FAKE_DOCKER_STATE.unready exists no container answers, so a test can hold a deploy (or its rollback)
# in wait_ready.
FAKE_CURL = r'''
import json, os, sys
if os.path.exists(os.environ["FAKE_DOCKER_STATE"] + ".unready"):
    sys.exit(7)
with open(os.environ["FAKE_DOCKER_STATE"]) as f:
    state = json.load(f)
ready = any(c["status"] == "running" and c["image"] not in state["unready_images"] for c in state["containers"].values())
sys.exit(0 if ready else 7)
'''

FAKE_SMOKE = r'''
import json, os, sys
with open(os.environ["FAKE_DOCKER_STATE"]) as f:
    state = json.load(f)
image = state["containers"][os.environ["CONTAINER_NAME"]]["image"]
print(f"smoke test of {os.environ['CONTAINER_NAME']} ({image})")
sys.exit(1 if image in state["smoke_fail_images"] else 0)
'''


def _script(path: Path, body: str, interpreter: str | None = None):
    path.write_text((f"#!{interpreter}\n" if interpreter else "") + body.lstrip("\n"), encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def deploy(tmp_path):
    """A project checkout (the real run.sh, stand-in patchers and smoke test), a host root holding a running
    deployment of OLD_ID, and stand-in tools. Returns run(env) -> CompletedProcess, plus accessors."""
    project = tmp_path / "project"
    (project / "gb10").mkdir(parents=True)
    shutil.copy2(ROOT / "gb10" / "run.sh", project / "gb10" / "run.sh")
    for patcher in (ROOT / "gb10").glob("patch-*.py"):
        _script(project / "gb10" / patcher.name, "import sys\nsys.exit(0)\n", sys.executable)
    _script(project / "gb10" / "smoke-test.sh", FAKE_SMOKE, sys.executable)
    (project / "extensions" / "ext-a").mkdir(parents=True)
    (project / "extensions" / "ext-a" / "main.py").write_text("new\n", encoding="utf-8")
    (project / "extensions" / "ext-new" / "scripts").mkdir(parents=True)
    (project / "extensions" / "ext-new" / "scripts" / "s.py").write_text("added\n", encoding="utf-8")

    host = tmp_path / "host"
    ext_a = host / "Extensions" / "ext-a"
    (ext_a / "data").mkdir(parents=True)
    (ext_a / "main.py").write_text("old\n", encoding="utf-8")
    (ext_a / "stale.py").write_text("removed from the checkout\n", encoding="utf-8")
    (ext_a / "data" / "saved.json").write_text("{}\n", encoding="utf-8")
    (host / "Extensions" / "ultimate-upscale-for-automatic1111").mkdir()
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "docker", FAKE_DOCKER, sys.executable)
    _script(bin_dir / "sudo", FAKE_SUDO)
    _script(bin_dir / "rsync", FAKE_RSYNC, sys.executable)
    _script(bin_dir / "curl", FAKE_CURL, sys.executable)
    (tmp_path / "tmp").mkdir()

    state_path = tmp_path / "docker-state.json"
    state = {
        "tags": {"local/gb10-a1111:latest": NEW_ID, "local/gb10-a1111:old": UNLABELED_ID},
        "images": {NEW_ID: NEW_LABELS, OLD_ID: {}, UNLABELED_ID: {"org.opencontainers.image.version": "24.04"}},
        "containers": {"gb10-a1111-latest": {"image": OLD_ID, "status": "running", "restarts": 0, "env": ["A1111_PORT=7860"]}},
        "unready_images": [], "crash_images": [], "smoke_fail_images": [], "calls": [],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    class Deploy:
        def __init__(self):
            self.host, self.project, self.tmp = host, project, tmp_path / "tmp"

        def set(self, **changes):
            current = self.state()
            current.update(changes)
            state_path.write_text(json.dumps(current), encoding="utf-8")

        def state(self):
            return json.loads(state_path.read_text(encoding="utf-8"))

        def sudo_commands(self):
            return (tmp_path / "sudo.log").read_text(encoding="utf-8").splitlines()

        def env(self, **env):
            full_env = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "FAKE_DOCKER_STATE": str(state_path),
                "FAKE_SUDO_LOG": str(tmp_path / "sudo.log"),
                "DOCKER_BIN": str(bin_dir / "docker"),
                "HOST_ROOT": str(host),
                "OUTPUTS_TARGET": str(outputs),
                "OPENCLAW_COMPILE_CACHE_NAMESPACE": "test-stack",
                "TMPDIR": str(tmp_path / "tmp"),
                "READY_TIMEOUT": "60",
                **env,
            }
            for name in ("IMAGE_TAG", "A1111_COMMIT_HASH", "A1111_VERSION_TAG", "COMMANDLINE_ARGS"):
                if name not in env:
                    full_env.pop(name, None)
            return full_env

        def run(self, **env):
            result = subprocess.run(["bash", str(project / "gb10" / "run.sh")], env=self.env(**env), capture_output=True, text=True, timeout=120)
            # stdout and stderr both go to the caller's stdout and to the deploy log.
            assert result.stderr == ""
            first, rest = result.stdout.split("\n", 1)
            assert first.startswith(f"Deploy log: {host}/deploy-logs/run-") and first.endswith(".log")
            assert Path(first.removeprefix("Deploy log: ")).read_text(encoding="utf-8") == rest
            return result

        def popen(self, **env):
            """run.sh in its own session (process group), its output on pipes the test may close."""
            return subprocess.Popen(["bash", str(project / "gb10" / "run.sh")], env=self.env(**env), stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

        def calls(self):
            """The docker calls so far, read while a deploy runs (the stand-in rewrites its state file)."""
            while True:
                try:
                    return self.state()["calls"]
                except json.JSONDecodeError:
                    time.sleep(0.05)

        def hold_unready(self):
            """While the returned file exists, no container answers the API."""
            hold = Path(str(state_path) + ".unready")
            hold.touch()
            return hold

    return Deploy()


def _calls(state, command):
    return [call for call in state["calls"] if call[0] == command]


def _assert_rolled_back(deploy, result):
    assert result.returncode != 0
    assert "rolling back" in result.stdout
    assert "Rolled back: gb10-a1111-latest runs the previous image" in result.stdout
    state = deploy.state()
    assert state["containers"] == {"gb10-a1111-latest": {"image": OLD_ID, "status": "running", "restarts": 0, "env": ["A1111_PORT=7860"]}}
    ext_a = deploy.host / "Extensions" / "ext-a"
    assert (ext_a / "main.py").read_text(encoding="utf-8") == "old\n"
    assert (ext_a / "stale.py").exists()
    assert (ext_a / "data" / "saved.json").exists()
    assert not (deploy.host / "Extensions" / "ext-new").exists()
    assert list(deploy.tmp.iterdir()) == []


def test_a_healthy_image_replaces_the_running_container(deploy):
    result = deploy.run()

    assert result.returncode == 0, result.stdout
    state = deploy.state()
    assert list(state["containers"]) == ["gb10-a1111-latest"]
    new = state["containers"]["gb10-a1111-latest"]
    assert new["image"] == NEW_ID and new["status"] == "running"
    # Run by image ID, reporting the version the image's labels record.
    assert new["run_args"][-1] == NEW_ID
    assert "A1111_COMMIT_HASH=c0ffee" in new["env"]
    assert f"A1111_VERSION_TAG={NEW_LABELS['org.opencontainers.image.version']}" in new["env"]
    assert "PYTORCH_ALLOC_CONF=expandable_segments:True" in new["env"]
    # The old container was stopped gracefully and set aside, then removed once the new one passed.
    assert ["stop", "-t", "120", "gb10-a1111-latest"] in state["calls"]
    assert ["rename", "gb10-a1111-latest", "gb10-a1111-latest-previous"] in state["calls"]
    assert ["rm", "gb10-a1111-latest-previous"] in state["calls"]
    assert not any(call[:2] == ["rm", "-f"] for call in state["calls"])
    assert "smoke test of gb10-a1111-latest" in result.stdout
    ext_a = deploy.host / "Extensions" / "ext-a"
    assert (ext_a / "main.py").read_text(encoding="utf-8") == "new\n"
    assert not (ext_a / "stale.py").exists()
    assert (ext_a / "data" / "saved.json").exists()
    assert (deploy.host / "Extensions" / "ext-new" / "scripts" / "s.py").exists()
    # The app cache is a host mount, with the compile cache nested inside it.
    mounts = [new["run_args"][i + 1] for i, arg in enumerate(new["run_args"]) if arg == "-v"]
    assert f"{deploy.host}/Caches/app:/opt/stable-diffusion-webui/cache" in mounts
    assert f"{deploy.host}/Caches/compile:/opt/stable-diffusion-webui/cache/compile" in mounts
    assert list(deploy.tmp.iterdir()) == []
    # Host paths written as root are also tested as root.
    sudo_commands = deploy.sudo_commands()
    for check in (f"test -e {deploy.host}/config/config.json", f"test -d {deploy.host}/Extensions/ext-a",
                  f"test -d {deploy.host}/Extensions/ultimate-upscale-for-automatic1111", f"test -L {deploy.host}/Outputs"):
        assert check in sudo_commands


def test_an_outputs_symlink_to_an_unmounted_target_is_kept(deploy, tmp_path):
    """A dangling Outputs link (the NAS not mounted yet) to the expected target is the expected link, not a conflict."""
    target = tmp_path / "unmounted"
    (deploy.host / "Outputs").symlink_to(target)

    result = deploy.run(OUTPUTS_TARGET=str(target))

    assert result.returncode == 0, result.stdout
    assert os.readlink(deploy.host / "Outputs") == str(target)


def test_a_container_that_dies_during_startup_is_replaced_by_the_previous_one(deploy):
    deploy.set(crash_images=[NEW_ID])

    result = deploy.run()

    _assert_rolled_back(deploy, result)
    assert "stopped or restarted before it was ready" in result.stdout
    assert "fake container log line" in result.stdout
    assert f"docker tag {OLD_ID} local/gb10-a1111:latest" in result.stdout


def test_a_failed_smoke_test_rolls_back(deploy):
    deploy.set(smoke_fail_images=[NEW_ID])

    result = deploy.run()

    _assert_rolled_back(deploy, result)


def test_an_api_that_never_answers_rolls_back_after_the_timeout(deploy):
    deploy.set(unready_images=[NEW_ID])

    result = deploy.run(READY_TIMEOUT="1")

    _assert_rolled_back(deploy, result)
    assert "did not answer /sdapi/v1/progress on port 7860 within 1 s" in result.stdout


def test_a_failure_with_nothing_to_roll_back_to_removes_the_new_container(deploy):
    deploy.set(containers={}, crash_images=[NEW_ID])

    result = deploy.run()

    assert result.returncode != 0
    assert "No container was running before this deploy" in result.stdout
    assert deploy.state()["containers"] == {}


def test_a_leftover_previous_container_stops_the_deploy_before_anything_changes(deploy):
    state = deploy.state()
    state["containers"]["gb10-a1111-latest-previous"] = {"image": OLD_ID, "status": "exited", "restarts": 0, "env": []}
    deploy.set(containers=state["containers"])

    result = deploy.run()

    assert result.returncode == 1
    assert "an earlier deploy did not finish" in result.stdout
    assert not _calls(deploy.state(), "stop") and not _calls(deploy.state(), "run")
    assert (deploy.host / "Extensions" / "ext-a" / "main.py").read_text(encoding="utf-8") == "old\n"


def test_an_image_without_provenance_labels_needs_its_version_given(deploy):
    result = deploy.run(IMAGE_TAG="local/gb10-a1111:old")

    assert result.returncode == 1
    assert "carries no provenance labels" in result.stdout
    assert not _calls(deploy.state(), "stop")

    result = deploy.run(IMAGE_TAG="local/gb10-a1111:old", A1111_COMMIT_HASH="c22a9794", A1111_VERSION_TAG="v-old")

    assert result.returncode == 0, result.stdout
    env = deploy.state()["containers"]["gb10-a1111-latest"]["env"]
    assert "A1111_COMMIT_HASH=c22a9794" in env and "A1111_VERSION_TAG=v-old" in env


def test_stop_sh_stops_gracefully_before_removing(tmp_path, deploy):
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "FAKE_DOCKER_STATE": str(tmp_path / "docker-state.json"), "DOCKER_BIN": str(tmp_path / "bin" / "docker")}

    result = subprocess.run(["bash", str(ROOT / "gb10" / "stop.sh")], env=env, capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    state = deploy.state()
    assert state["containers"] == {}
    assert [call for call in state["calls"] if call[0] in ("stop", "rm")] == [["stop", "-t", "120", "gb10-a1111-latest"], ["rm", "gb10-a1111-latest"]]


def _wait_until(proc, condition, what):
    deadline = time.monotonic() + 60
    while not condition():
        assert proc.poll() is None and time.monotonic() < deadline, f"run.sh ended or timed out before {what}"
        time.sleep(0.1)


def _deploy_log(proc):
    return Path(proc.stdout.readline().decode().removeprefix("Deploy log: ").strip())


@pytest.mark.parametrize("smoke_fails", [False, True], ids=["deploys", "rolls-back"])
def test_a_lost_caller_does_not_stop_the_deploy_or_its_rollback(deploy, smoke_fails):
    """The caller's output pipes close and the session gets SIGHUP (an SSH drop, a caller killed on its timeout) while
    the new container starts: the deploy, or its rollback, still runs to the end, and the log holds all of it."""
    if smoke_fails:
        deploy.set(smoke_fail_images=[NEW_ID])
    hold = deploy.hold_unready()
    proc = deploy.popen()
    log = _deploy_log(proc)
    _wait_until(proc, lambda: any(call[0] == "run" for call in deploy.calls()), "the new container was started")
    proc.stdout.close()
    proc.stderr.close()
    os.killpg(proc.pid, signal.SIGHUP)
    time.sleep(0.5)
    hold.unlink()

    returncode = proc.wait(timeout=120)

    output = log.read_text(encoding="utf-8")
    assert "smoke test of gb10-a1111-latest" in output
    if smoke_fails:
        _assert_rolled_back(deploy, subprocess.CompletedProcess(proc.args, returncode, output, ""))
    else:
        assert returncode == 0, output
        assert output.rstrip().endswith("this image is API/headless only.")
        assert deploy.state()["containers"]["gb10-a1111-latest"]["image"] == NEW_ID


def test_ctrl_c_during_the_rollback_does_not_stop_it(deploy):
    """The new container never answers. While the rollback waits for the restarted previous container, Ctrl-C twice
    (SIGINT to the whole process group): the rollback still finishes, and the deploy exits with its first failure."""
    hold = deploy.hold_unready()
    proc = deploy.popen(READY_TIMEOUT="8")
    log = _deploy_log(proc)
    _wait_until(proc, lambda: ["start", "gb10-a1111-latest"] in deploy.calls(), "the rollback restarted the previous container")
    for _ in range(2):
        os.killpg(proc.pid, signal.SIGINT)
        time.sleep(0.5)
    hold.unlink()

    out, err = proc.communicate(timeout=120)

    output = log.read_text(encoding="utf-8")
    assert out.decode() == output and err == b""
    _assert_rolled_back(deploy, subprocess.CompletedProcess(proc.args, proc.returncode, output, ""))
    assert proc.returncode == 1
    assert "within 8 s" in output and "ERROR: rollback:" not in output
