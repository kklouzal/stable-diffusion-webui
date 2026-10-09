from __future__ import annotations
from typing import List, NamedTuple, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from transformers.models.clip.modeling_clip import CLIPVisionModelOutput

from .image_proj_models import (
    Resampler,
    ImageProjModel,
    MLPProjModel,
    MLPProjModelFaceId,
)


class ImageEmbed(NamedTuple):
    """Image embed for a single image."""

    cond_emb: torch.Tensor
    uncond_emb: torch.Tensor

    def average_of(*args: List[Tuple[torch.Tensor, torch.Tensor]]) -> "ImageEmbed":
        conds, unconds = zip(*args)

        def average_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.sum(torch.stack(tensors), dim=0) / len(tensors)

        return ImageEmbed(average_tensors(conds), average_tensors(unconds))


class To_KV(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            k = key.replace(".weight", "").replace(".", "_")
            self.to_kvs[k] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[k].weight.data = value


class IPAdapterModel(torch.nn.Module):
    def __init__(
        self,
        state_dict,
        clip_embeddings_dim,
        cross_attention_dim,
        is_plus,
        is_sdxl: bool,
        sdxl_plus,
        is_full,
        is_faceid: bool,
        is_portrait: bool,
        is_instantid: bool,
    ):
        super().__init__()
        self.device = "cpu"

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.is_plus = is_plus
        self.is_sdxl = is_sdxl
        self.sdxl_plus = sdxl_plus
        self.is_full = is_full
        self.is_faceid = is_faceid
        self.is_instantid = is_instantid
        self.clip_extra_context_tokens = 16 if (self.is_plus or is_portrait) else 4

        if self.is_instantid:
            self.image_proj_model = self.init_proj_instantid()
        elif is_faceid:
            self.image_proj_model = self.init_proj_faceid()
        elif self.is_plus:
            if self.is_full:
                self.image_proj_model = MLPProjModel(
                    cross_attention_dim=cross_attention_dim,
                    clip_embeddings_dim=clip_embeddings_dim,
                )
            else:
                self.image_proj_model = Resampler(
                    dim=1280 if sdxl_plus else cross_attention_dim,
                    depth=4,
                    dim_head=64,
                    heads=20 if sdxl_plus else 12,
                    num_queries=self.clip_extra_context_tokens,
                    embedding_dim=clip_embeddings_dim,
                    output_dim=self.cross_attention_dim,
                    ff_mult=4,
                )
        else:
            self.clip_extra_context_tokens = (
                state_dict["image_proj"]["proj.weight"].shape[0]
                // self.cross_attention_dim
            )

            self.image_proj_model = ImageProjModel(
                cross_attention_dim=self.cross_attention_dim,
                clip_embeddings_dim=clip_embeddings_dim,
                clip_extra_context_tokens=self.clip_extra_context_tokens,
            )

        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = To_KV(state_dict["ip_adapter"])

    def init_proj_faceid(self):
        return MLPProjModelFaceId(
            cross_attention_dim=self.cross_attention_dim,
            id_embeddings_dim=512,
            num_tokens=self.clip_extra_context_tokens,
        )

    def init_proj_instantid(self, image_emb_dim=512, num_tokens=16):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=self.cross_attention_dim,
            ff_mult=4,
        )
        return image_proj_model

    @torch.inference_mode()
    def _get_image_embeds(
        self, clip_vision_output: CLIPVisionModelOutput
    ) -> ImageEmbed:
        self.image_proj_model.to(self.device)

        if self.is_plus:
            from annotator.clipvision import clip_vision_h_uc

            cond = self.image_proj_model(
                clip_vision_output["hidden_states"][-2].to(
                    device=self.device, dtype=torch.float32
                )
            )
            # This model's projection of the ViT-H penultimate hidden states of a zero image, for SD1.5 and
            # SDXL alike (IPAdapterPlus/IPAdapterPlusXL.get_image_embeds of tencent-ailab/IP-Adapter): a fixed
            # projected uncond would belong to one SDXL plus checkpoint only.
            uncond = self.image_proj_model(clip_vision_h_uc.to(cond))
            return ImageEmbed(cond, uncond)

        clip_image_embeds = clip_vision_output["image_embeds"].to(
            device=self.device, dtype=torch.float32
        )
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        # input zero vector for unconditional.
        uncond_image_prompt_embeds = self.image_proj_model(
            torch.zeros_like(clip_image_embeds)
        )
        return ImageEmbed(image_prompt_embeds, uncond_image_prompt_embeds)

    @torch.inference_mode()
    def _get_image_embeds_faceid(self, insightface_output: torch.Tensor) -> ImageEmbed:
        """Get image embeds for non-plus faceid. Multiple inputs are supported."""
        self.image_proj_model.to(self.device)
        faceid_embed = insightface_output.to(self.device, dtype=torch.float32)
        return ImageEmbed(
            self.image_proj_model(faceid_embed),
            self.image_proj_model(torch.zeros_like(faceid_embed)),
        )

    @torch.inference_mode()
    def _get_image_embeds_instantid(
        self, prompt_image_emb: Union[torch.Tensor, np.ndarray]
    ) -> ImageEmbed:
        """Get image embeds for instantid."""
        image_proj_model_in_features = 512
        if isinstance(prompt_image_emb, torch.Tensor):
            prompt_image_emb = prompt_image_emb.clone().detach()
        else:
            prompt_image_emb = torch.tensor(prompt_image_emb)

        prompt_image_emb = prompt_image_emb.to(device=self.device, dtype=torch.float32)
        prompt_image_emb = prompt_image_emb.reshape(
            [1, -1, image_proj_model_in_features]
        )
        return ImageEmbed(
            self.image_proj_model(prompt_image_emb),
            self.image_proj_model(torch.zeros_like(prompt_image_emb)),
        )

    @staticmethod
    def load(state_dict: dict, model_name: str) -> IPAdapterModel:
        """
        Arguments:
            - state_dict: model state_dict.
            - model_name: file name of the model.

        PuLID and FaceID Plus models are rejected: their image embeds come only from the ip-adapter_pulid and
        ip-adapter_face_id_plus preprocessors, which this build does not ship (and `ipadapter_input` decodes
        plain tensors only).
        """
        is_faceid = "faceid" in model_name
        is_instantid = "instant_id" in model_name
        is_portrait = "portrait" in model_name
        is_full = "proj.3.weight" in state_dict["image_proj"]
        is_plus = (
            is_full
            or "latents" in state_dict["image_proj"]
            or "perceiver_resampler.proj_in.weight" in state_dict["image_proj"]
        )
        if "pulid" in model_name.lower():
            raise RuntimeError(
                f"[ControlNet Error] {model_name}: PuLID IP-Adapter models need the removed ip-adapter_pulid "
                "preprocessor and are not supported by this build.")
        if is_faceid and is_plus:
            raise RuntimeError(
                f"[ControlNet Error] {model_name}: FaceID Plus IP-Adapter models need the removed "
                "ip-adapter_face_id_plus preprocessor and are not supported by this build.")
        cross_attention_dim = state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        sdxl = cross_attention_dim == 2048
        sdxl_plus = sdxl and is_plus

        if is_instantid or is_faceid:
            # InstantID and plain FaceID do not use clip embedding.
            clip_embeddings_dim = None
        elif is_plus:
            if sdxl_plus:
                clip_embeddings_dim = int(state_dict["image_proj"]["latents"].shape[2])
            elif is_full:
                clip_embeddings_dim = int(
                    state_dict["image_proj"]["proj.0.weight"].shape[1]
                )
            else:
                clip_embeddings_dim = int(
                    state_dict["image_proj"]["proj_in.weight"].shape[1]
                )
        else:
            clip_embeddings_dim = int(state_dict["image_proj"]["proj.weight"].shape[1])

        return IPAdapterModel(
            state_dict,
            clip_embeddings_dim=clip_embeddings_dim,
            cross_attention_dim=cross_attention_dim,
            is_plus=is_plus,
            is_sdxl=sdxl,
            sdxl_plus=sdxl_plus,
            is_full=is_full,
            is_faceid=is_faceid,
            is_portrait=is_portrait,
            is_instantid=is_instantid,
        )

    def get_image_emb(self, preprocessor_output) -> ImageEmbed:
        if self.is_instantid:
            return self._get_image_embeds_instantid(preprocessor_output)
        elif self.is_faceid:
            return self._get_image_embeds_faceid(preprocessor_output)
        else:
            return self._get_image_embeds(preprocessor_output)
