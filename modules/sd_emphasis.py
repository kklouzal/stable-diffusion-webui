from __future__ import annotations
import torch


class Emphasis:
    """Emphasis class decides how to death with (emphasized:1.1) text in prompts"""

    name: str = "Base"
    description: str = ""

    tokens: list[list[int]]
    """tokens from the chunk of the prompt"""

    multipliers: torch.Tensor
    """tensor with multipliers, once for each token"""

    z: torch.Tensor
    """output of cond transformers network (CLIP)"""

    def after_transformers(self):
        """Called after cond transformers network has processed the chunk of the prompt; this function should modify self.z to apply the emphasis"""

        pass


class EmphasisNone(Emphasis):
    name = "None"
    description = "disable the mechanism entirely and treat (:.1.1) as literal characters"


class EmphasisIgnore(Emphasis):
    name = "Ignore"
    description = "treat all empasised words as if they have no emphasis"


class EmphasisOriginal(Emphasis):
    name = "Original"
    description = "the original emphasis implementation"

    def after_transformers(self):
        # In float32: a bf16 original_mean is rounded (up to 2**-9 relative), which rescaled every chunk even when
        # all multipliers are 1. The result was already float32 through promotion with the float32 multipliers.
        z = self.z.float()
        original_mean = z.mean()
        z = z * self.multipliers.reshape(self.multipliers.shape + (1,))

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        new_mean = z.mean()
        self.z = z * (original_mean / new_mean)


class EmphasisOriginalNoNorm(EmphasisOriginal):
    name = "No norm"
    description = "same as original, but without normalization (seems to work better for SDXL)"

    def after_transformers(self):
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)


def get_current_option(emphasis_option_name):
    return next(iter([x for x in options if x.name == emphasis_option_name]), EmphasisOriginal)


def get_options_descriptions():
    return ", ".join(f"{x.name}: {x.description}" for x in options)


options = [
    EmphasisNone,
    EmphasisIgnore,
    EmphasisOriginal,
    EmphasisOriginalNoNorm,
]
