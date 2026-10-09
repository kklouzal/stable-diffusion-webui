import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DPT_DINOv2, DPTHead
from depth_anything_v2.dinov2 import DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from .util import load_model
from .annotator_path import models_path


transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)


class DepthAnythingV1(DPT_DINOv2):
    """Depth Anything v1 (DPT_DINOv2, ViT-L/14 encoder) with the DINOv2 backbone depth_anything_v2 vendors.

    DPT_DINOv2.__init__ gets its backbone from torch.hub: facebookresearch/dinov2 fetched from GitHub (unpinned remote
    code, plus an ImageNet weight download the checkpoint then overwrites), or a local hub checkout this install does not
    have. depth_anything_v2's DINOv2("vitl") builds the hub's dinov2_vitl14 architecture (img_size 518, patch 14, layer
    scale 1.0, MLP FFN, no block chunks, no register tokens, interpolate offset 0.1) with the same parameter names, so
    the v1 checkpoint loads strictly. forward() is DPT_DINOv2's (the last 4 blocks feed the DPT head).
    """

    def __init__(self):
        torch.nn.Module.__init__(self)  # DPT_DINOv2.__init__ would call torch.hub.load
        self.pretrained = DINOv2(model_name="vitl")
        self.depth_head = DPTHead(
            1, self.pretrained.embed_dim, features=256, use_bn=False,
            out_channels=[256, 512, 1024, 1024], use_clstoken=False,
        )


class DepthAnythingDetector:
    """https://github.com/LiheYoung/Depth-Anything"""

    model_dir = os.path.join(models_path, "depth_anything")

    def __init__(self, device: torch.device):
        self.device = device
        self.model = DepthAnythingV1().to(device).eval()
        remote_url = os.environ.get(
            "CONTROLNET_DEPTH_ANYTHING_MODEL_URL",
            "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth",
        )
        model_path = load_model(
            "depth_anything_vitl14.pth", remote_url=remote_url, model_dir=self.model_dir
        )
        self.model.load_state_dict(torch.load(model_path), strict=True)

    def __call__(self, image: np.ndarray, colored: bool = True) -> np.ndarray:
        self.model.to(self.device)
        h, w = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        @torch.no_grad()
        def predict_depth(model, image):
            return model(image)
        depth = predict_depth(self.model, image)
        depth = F.interpolate(
            depth[None], (h, w), mode="bilinear", align_corners=False
        )[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        if colored:
            return cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        else:
            return depth

    def unload_model(self):
        self.model.to("cpu")
