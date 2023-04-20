import os
import cv2
import numpy as np
from PIL import Image
import torch

from huggingface_hub import hf_hub_download
from einops import rearrange
from .zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoedepth.utils.config import get_config
from ..util import HWC3

class ZoeDetector:
    def __init__(self, model):
        self.model = model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model.device = 'cuda'

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None):
        filename = filename or "ZoeD_M12_N.pt"

        model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)
        conf = get_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])

        return cls(model)

    def __call__(self, input_image):
        input_type = "np"
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
            input_type = "pil"

        input_image = HWC3(input_image)
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float()
            if torch.cuda.is_available():
                image_depth = image_depth.cuda()
            image_depth = image_depth / 255.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model.infer(image_depth)

            depth = depth[0, 0].cpu().numpy()

            vmin = np.percentile(depth, 2)
            vmax = np.percentile(depth, 85)

            depth -= vmin
            depth /= vmax - vmin
            depth = 1.0 - depth
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

        if input_type == "pil":
            depth_image = Image.fromarray(depth_image)
            depth_image = depth_image.convert("RGB")

        return depth_image
