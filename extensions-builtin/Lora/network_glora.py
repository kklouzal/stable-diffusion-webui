
import torch

import network

class ModuleTypeGLora(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["a1.weight", "a2.weight", "alpha", "b1.weight", "b2.weight"]):
            return NetworkModuleGLora(net, weights)

        return None

# adapted from https://github.com/KohakuBlueleaf/LyCORIS; layout detection as in ComfyUI weight_adapter/glora.py
class NetworkModuleGLora(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)

        if hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape

        self.w1a = weights.w["a1.weight"]
        self.w1b = weights.w["b1.weight"]
        self.w2a = weights.w["a2.weight"]
        self.w2b = weights.w["b2.weight"]

        # Old LyCORIS layout: a1 [r, in], a2 [in, r], b1 [r, in], b2 [out, r]; dW = b2 b1 + W a2 a1.
        # Current layout (LyCORIS 2024-05+): a1 [in, r], a2 [r, in], b1 [out, r], b2 [r, in*k*k]; dW = W a1 a2 + b1 b2.
        # Both scale dW by alpha / r.
        a1, a2, b1, b2 = self.w1a, self.w2a, self.w1b, self.w2b
        old = b2.shape[1] == b1.shape[0] == a1.shape[0] == a2.shape[1]
        new = b2.shape[0] == b1.shape[1] == a1.shape[1] == a2.shape[0]
        if old and new:
            old = a2.shape[0] == self.shape[0] == self.shape[1]
        self.old_layout = old or not new
        self.dim = a1.shape[0] if self.old_layout else a2.shape[0]

    def calc_updown(self, orig_weight):
        a1 = self.w1a.to(orig_weight.device, dtype=torch.float32).flatten(1)
        b1 = self.w1b.to(orig_weight.device, dtype=torch.float32).flatten(1)
        a2 = self.w2a.to(orig_weight.device, dtype=torch.float32).flatten(1)
        b2 = self.w2b.to(orig_weight.device, dtype=torch.float32).flatten(1)
        weight = orig_weight.to(torch.float32)

        if self.old_layout:
            updown = b2 @ b1 + (weight.flatten(1) @ a2) @ a1
        elif weight.dim() > 2:
            updown = torch.einsum("o i ..., i j -> o j ...", torch.einsum("o i ..., i j -> o j ...", weight, a1), a2) + (b1 @ b2).reshape(weight.shape)
        else:
            updown = (weight @ a1) @ a2 + b1 @ b2

        return self.finalize_updown(updown.reshape(orig_weight.shape), orig_weight, orig_weight.shape)
