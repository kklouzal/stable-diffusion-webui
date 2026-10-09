import os
import sys
from pathlib import Path


# setup_test_env
os.environ['IGNORE_CMD_ARGS_ERRORS'] = 'True'

file_path = Path(__file__).resolve()
ext_root = file_path.parent.parent
a1111_root = ext_root.parent.parent

for p in (ext_root, a1111_root):
    if p not in sys.path:
        sys.path.append(str(p))

# Initialize A1111, but without the startup model load thread: while it runs, sd_models.load_model patches
# torch.nn layer constructors, init functions and load_state_dict process-wide (sd_disable_initialization), so
# torch modules that tests in this process build meanwhile would get meta or uninitialized parameters. A test
# that needs the checkpoint loads it on its first shared.sd_model access, in its own thread. The tests run on the
# CUDA device when there is one and on the CPU otherwise, which startup refuses without --skip-torch-cuda-test.
from modules import initialize, launch_utils  # noqa: E402
launch_utils.args.skip_torch_cuda_test = True
initialize.imports()
from modules import shared  # noqa: E402
_skip_load_model_at_start = shared.cmd_opts.skip_load_model_at_start
shared.cmd_opts.skip_load_model_at_start = True
try:
    initialize.initialize()
finally:
    shared.cmd_opts.skip_load_model_at_start = _skip_load_model_at_start

