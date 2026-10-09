import ast
from pathlib import Path

import pytest
import torch

from modules.models.diffusion.uni_pc import uni_pc


@pytest.mark.parametrize("path", ["modules/sd_samplers_kdiffusion.py", "modules/sd_samplers_timesteps.py"])
def test_sampler_functions_get_the_console_progressbar_flag(path):
    tree = ast.parse(Path(path).read_text(encoding="utf8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "self.func"]

    assert len(calls) == 2  # txt2img sample() and img2img sample_img2img()
    for call in calls:
        disable = [kw.value for kw in call.keywords if kw.arg == "disable"]
        assert [ast.unparse(value) for value in disable] == ["shared.cmd_opts.disable_console_progressbars"]


@pytest.mark.parametrize("disable", [True, False])
def test_unipc_progress_bar_honors_disable(capsys, disable):
    alphas_cumprod = torch.cumprod(1 - torch.linspace(0.00085, 0.012, 1000, dtype=torch.float64), 0)
    sampler = uni_pc.UniPC(lambda x, t, cond, uncond: x * 0.5, uni_pc.NoiseScheduleVP("discrete", alphas_cumprod=alphas_cumprod))

    sampler.sample(torch.ones(1, 4, 2, 2, dtype=torch.float64), steps=4, order=2, disable=disable)

    assert ("4/4" in capsys.readouterr().err) is not disable
