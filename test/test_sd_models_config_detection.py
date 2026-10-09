import pytest
import torch

from modules import sd_models_config as config

UNET_IN = "model.diffusion_model.input_blocks.0.0.weight"
SDXL_CLIP_G = "conditioner.embedders.1.model.ln_final.weight"


@pytest.mark.parametrize("state_dict,expected", [
    ({"model.diffusion_model.x_embedder.proj.weight": torch.empty(1)}, "config_sd3"),
    ({SDXL_CLIP_G: torch.empty(1), UNET_IN: torch.empty(320, 4, 3, 3)}, "config_sdxl"),
    ({SDXL_CLIP_G: torch.empty(1), UNET_IN: torch.empty(320, 9, 3, 3)}, "config_sdxl_inpainting"),
    ({SDXL_CLIP_G: torch.empty(1), UNET_IN: torch.empty(320, 4, 3, 3), "v_pred": torch.empty(())}, "config_sdxlv"),
    ({"conditioner.embedders.0.model.ln_final.weight": torch.empty(1), UNET_IN: torch.empty(384, 4, 3, 3)}, "config_sdxl_refiner"),
    ({"cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight": torch.empty(3, 1024), UNET_IN: torch.empty(320, 9, 3, 3)}, "config_sd2_inpainting"),
    ({UNET_IN: torch.empty(320, 4, 3, 3)}, "config_default"),
    ({UNET_IN: torch.empty(320, 9, 3, 3)}, "config_inpainting"),
    ({UNET_IN: torch.empty(320, 8, 3, 3)}, "config_instruct_pix2pix"),
    # Alt-Diffusion support was removed: its XLM-R text encoder keys no longer select a config of their own.
    ({UNET_IN: torch.empty(320, 4, 3, 3), "cond_stage_model.roberta.embeddings.word_embeddings.weight": torch.empty(8, 4),
      "cond_stage_model.transformation.weight": torch.empty(8, 4)}, "config_default"),
])
def test_guess_model_config_from_state_dict_by_family(state_dict, expected):
    assert config.guess_model_config_from_state_dict(dict(state_dict), "model.safetensors") == getattr(config, expected)
