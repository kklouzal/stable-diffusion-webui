import os

import torch

from modules import shared
from modules.shared import cmd_opts


def initialize():
    """Initializes fields inside the shared module in a controlled manner.

    Should be called early because some other modules you can import mingt need these fields to be already set.
    """

    os.makedirs(cmd_opts.hypernetwork_dir, exist_ok=True)

    from modules import options, shared_options
    shared.options_templates = shared_options.options_templates
    shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)
    shared.restricted_opts = shared_options.restricted_opts
    try:
        shared.opts.load(shared.config_filename)
    except FileNotFoundError:
        pass

    from modules import devices
    devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
        (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if cmd_opts.dtype == "auto":
        devices.dtype = torch.float32 if cmd_opts.no_half else torch.float16
    else:
        assert not cmd_opts.no_half or cmd_opts.dtype == "float32", "--no-half conflicts with --dtype values other than float32"
        devices.dtype = dtype_map[cmd_opts.dtype]

    if devices.dtype == torch.bfloat16 and devices.device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--dtype bfloat16 requires a CUDA device with bfloat16 support")

    devices.dtype_vae = torch.float32 if cmd_opts.no_half_vae else devices.dtype
    devices.dtype_unet = devices.dtype
    devices.dtype_inference = torch.float32 if cmd_opts.precision == 'full' else devices.dtype

    if cmd_opts.precision == "half":
        msg = "--precision half requires fp16 model and VAE dtype; use --precision autocast with --dtype bfloat16"
        assert devices.dtype == torch.float16, msg
        assert devices.dtype_vae == torch.float16, msg
        assert devices.dtype_inference == torch.float16, msg
        devices.force_fp16 = True
        devices.force_model_fp16()

    shared.device = devices.device
    shared.weight_load_location = None if cmd_opts.lowram else "cpu"

    from modules import shared_state
    shared.state = shared_state.State()

    from modules import styles
    shared.prompt_styles = styles.StyleDatabase(shared.styles_filename)

    from modules import interrogate
    shared.interrogator = interrogate.InterrogateModels("interrogate")

    from modules import shared_total_tqdm
    shared.total_tqdm = shared_total_tqdm.TotalTQDM()

    from modules import memmon, devices
    shared.mem_mon = memmon.MemUsageMonitor("MemMon", devices.device, shared.opts)
    shared.mem_mon.start()

