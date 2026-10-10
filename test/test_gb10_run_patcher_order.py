"""gb10/run.sh ordering: everything that can fail runs while production still serves; the live container is stopped
only right before the owned-extension sync, the third-party patchers and the new container start."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUN_SH = (ROOT / "gb10" / "run.sh").read_text(encoding="utf-8")
TEARDOWN = 'sudo "${DOCKER_BIN}" stop -t "${STOP_TIMEOUT}" "${CONTAINER_NAME}"'


def test_every_patcher_runs_exactly_once_per_pass_and_every_referenced_patcher_exists():
    function = RUN_SH[RUN_SH.index("patch_third_party_extensions() {"):]
    function = function[:function.index("\n}\n")]
    invoked = re.findall(r'^\s*sudo python3 "\$\{PROJECT_ROOT\}/gb10/(patch-[\w-]+\.py)" "\$\{extensions_root\}/[\w-]+"$', function, flags=re.M)
    on_disk = sorted(path.name for path in (ROOT / "gb10").glob("patch-*.py"))
    assert sorted(invoked) == on_disk
    # Patchers run only through the function (rehearsal, then the real pass); an apply run already verifies an
    # already-patched target, so a separate --check run would re-prove the same bytes.
    assert len(re.findall(r"gb10/patch-[\w-]+\.py", RUN_SH)) == len(on_disk)
    assert not re.search(r'gb10/patch-[\w-]+\.py" --check', RUN_SH)
    md_guard = function.index('if sudo test -d "${extensions_root}/multidiffusion-upscaler-for-automatic1111"; then')
    assert md_guard < function.index("patch-multidiffusion-performance.py") < function.index("\n  fi\n", md_guard)
    assert function.index("patch-ultimate-upscale-state-lifecycle.py") < function.index("patch-ultimate-upscale-subcanvas.py")
    # Production requests use Detail Daemon: its patcher runs unconditionally, so a missing checkout fails the deploy.
    assert function.index("\n  fi\n", md_guard) < function.index("patch-detail-daemon.py")


def test_failable_steps_precede_the_teardown_and_mutations_follow_it():
    teardown = RUN_SH.rindex(TEARDOWN)  # the rollback function, defined earlier, stops the failed new container
    before = (
        "find \"${PROJECT_ROOT}/extensions\"",  # owned-extension discovery (fails on none)
        "--entrypoint python \"${TARGET_IMAGE_ID}\"",  # image compile-stack probe
        "cannot read the host NVIDIA driver version",
        "compile cache namespace is not writable",
        'image inspect "${IMAGE_TAG}"',
        "carries no provenance labels",
        "an earlier deploy did not finish",
        'patch_third_party_extensions "${PATCH_REHEARSAL_ROOT}"',
    )
    for step in before:
        assert RUN_SH.index(step) < teardown, step
    sync = RUN_SH.index('mirror_owned_extension "${PROJECT_ROOT}/extensions/${extension_name}"')
    patch = RUN_SH.index('patch_third_party_extensions "${HOST_ROOT}/Extensions"')
    chown = RUN_SH.index("sudo chown -R 2323:2323")
    start = RUN_SH.index('sudo "$DOCKER_BIN" run "${DOCKER_ARGS[@]}" "${TARGET_IMAGE_ID}"')
    assert teardown < sync < patch < chown < start
    # The rehearsal patches a scratch copy, never the live Extensions tree, and the scratch root goes on exit.
    rehearsal = RUN_SH[RUN_SH.index("PATCH_REHEARSAL_ROOT=") : teardown]
    assert 'SCRATCH_ROOT="$(mktemp -d' in RUN_SH and 'PATCH_REHEARSAL_ROOT="${SCRATCH_ROOT}/patch-rehearsal"' in rehearsal
    on_exit = RUN_SH[RUN_SH.index("on_exit() {"):]
    assert 'sudo rm -rf -- "${SCRATCH_ROOT}"' in on_exit[:on_exit.index("\n}\n")]
    assert '"${HOST_ROOT}/Extensions/${third_party_extension}" "${PATCH_REHEARSAL_ROOT}/"' in rehearsal
    # Every patched checkout is copied into the rehearsal root.
    rehearsed = re.search(r"^for third_party_extension in ([\w -]+); do$", rehearsal, flags=re.M).group(1).split()
    function = RUN_SH[RUN_SH.index("patch_third_party_extensions() {"):]
    patched = re.findall(r'"\$\{extensions_root\}/([\w-]+)"$', function[:function.index("\n}\n")], flags=re.M)
    assert sorted(rehearsed) == sorted(set(patched))
