import torch


class MemUsageMonitor:
    """Device memory queries for shared.mem_mon.

    Extensions read free/total device memory through cuda_mem_get_info() (multidiffusion-upscaler's tile scripts).
    disabled is True when the device cannot be queried (no CUDA device).
    """

    disabled = False

    def __init__(self, name, device, opts):
        self.name = name
        self.device = device
        self.opts = opts

        try:
            self.cuda_mem_get_info()
        except Exception as e:  # AMD or whatever
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    def cuda_mem_get_info(self):
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        return torch.cuda.mem_get_info(index)
