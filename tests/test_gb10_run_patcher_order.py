"""gb10/run.sh applies every deploy patcher after the owned-extension sync and before the container starts."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUN_SH = (ROOT / "gb10" / "run.sh").read_text(encoding="utf-8")


def test_every_patcher_runs_exactly_once_and_every_referenced_patcher_exists():
    invoked = re.findall(r'^\s*sudo python3 "\$\{PROJECT_ROOT\}/gb10/(patch-[\w-]+\.py)" "\$\{\w+\}"$', RUN_SH, flags=re.M)
    on_disk = sorted(path.name for path in (ROOT / "gb10").glob("patch-*.py"))
    assert sorted(invoked) == on_disk
    # An apply run already verifies an already-patched target; a separate --check run would re-prove the same bytes.
    assert not re.search(r'gb10/patch-[\w-]+\.py" --check', RUN_SH)


def test_patchers_run_after_the_extension_sync_and_before_the_container_starts():
    sync = RUN_SH.index("rsync -a ")
    md_block = RUN_SH.index('if [[ -d "${MULTIDIFFUSION_ROOT}" ]]; then')
    md_end = RUN_SH.index("\nfi\n", md_block)
    multidiffusion = RUN_SH.index("gb10/patch-multidiffusion-performance.py")
    lifecycle = RUN_SH.index("gb10/patch-ultimate-upscale-state-lifecycle.py")
    subcanvas = RUN_SH.index("gb10/patch-ultimate-upscale-subcanvas.py")
    container_start = RUN_SH.index('sudo "$DOCKER_BIN" run "${DOCKER_ARGS[@]}"')

    assert 'MULTIDIFFUSION_ROOT="${HOST_ROOT}/Extensions/multidiffusion-upscaler-for-automatic1111"' in RUN_SH
    assert 'ULTIMATE_UPSCALE_ROOT="${HOST_ROOT}/Extensions/ultimate-upscale-for-automatic1111"' in RUN_SH
    assert sync < md_block < multidiffusion < md_end < lifecycle < subcanvas < container_start
