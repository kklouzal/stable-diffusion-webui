import json
import os
import subprocess
import sys
from pathlib import Path

PROBE = """
import json, os, sys
os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:int(sys.argv[1])]))
import torch
defaults = [torch.get_num_threads(), torch.get_num_interop_threads()]
from modules import initialize_util
initialize_util.configure_torch_threads()
print(json.dumps({"defaults": defaults, "configured": [torch.get_num_threads(), torch.get_num_interop_threads()]}))
"""


def _probe(cpus, **env):
    environ = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    environ.update(env)
    result = subprocess.run([sys.executable, "-c", PROBE, str(cpus)], env=environ, capture_output=True, text=True, check=True, timeout=300)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_threads_follow_the_affinity_mask():
    assert _probe(1)["configured"] == [1, 1]
    if len(os.sched_getaffinity(0)) >= 3:
        assert _probe(3)["configured"] == [3, 2]


def test_explicit_omp_num_threads_is_left_alone():
    result = _probe(1, OMP_NUM_THREADS="3")
    assert result["configured"] == result["defaults"]
    assert result["configured"][0] == 3


def test_thread_pools_are_sized_right_after_torch_import():
    source = Path("modules/initialize.py").read_text(encoding="utf8")
    imports = source[source.index("def imports():"):source.index("def initialize():")]
    assert imports.index("    import torch  # noqa: F401") < imports.index("initialize_util.configure_torch_threads()") < imports.index("    import pytorch_lightning")
