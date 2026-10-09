import json
import os
import subprocess
import sys

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


IMPORT_ORDER_PROBE = """
import json, sys
events = []

class StopAtLightning(Exception):
    pass

class LightningImportProbe:
    def find_spec(self, name, path=None, target=None):
        if name == "pytorch_lightning":
            events.append("import pytorch_lightning")
            raise StopAtLightning
        return None

sys.meta_path.insert(0, LightningImportProbe())
from modules import initialize, initialize_util
initialize_util.configure_torch_threads = lambda: events.append(["configure_torch_threads", "torch" in sys.modules])
try:
    initialize.imports()
except StopAtLightning:
    pass
print(json.dumps(events))
"""


def test_thread_pools_are_sized_right_after_torch_import():
    # Fresh interpreter: initialize.imports() up to the pytorch_lightning import, which the probe stops. The test host may
    # have no CUDA device, which startup refuses without --skip-torch-cuda-test.
    result = subprocess.run([sys.executable, "-c", IMPORT_ORDER_PROBE, "--skip-torch-cuda-test"], capture_output=True, text=True, check=True, timeout=300)
    events = json.loads(result.stdout.strip().splitlines()[-1])

    assert events == [["configure_torch_threads", True], "import pytorch_lightning"]
