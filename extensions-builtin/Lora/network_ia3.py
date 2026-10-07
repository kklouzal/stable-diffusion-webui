import torch

import network


class ModuleTypeIa3(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["weight"]):
            return NetworkModuleIa3(net, weights)

        return None


class NetworkModuleIa3(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)

        self.w = weights.w["weight"]
        self.on_input = weights.w["on_input"].item()

    def calc_updown(self, orig_weight):
        w = self.w.to(orig_weight.device, dtype=torch.float32)

        # LyCORIS stores one scale per input or output channel ([dim] for Linear, [1, dim, 1, 1] for Conv2d);
        # it multiplies weight axis 1 (input) or axis 0 (output).
        if self.on_input:
            w = w.reshape(1, -1, *[1] * (orig_weight.dim() - 2))
        else:
            w = w.reshape(-1, *[1] * (orig_weight.dim() - 1))

        updown = orig_weight.to(torch.float32) * w

        return self.finalize_updown(updown, orig_weight, orig_weight.shape)
