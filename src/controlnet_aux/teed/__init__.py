import os

import cv2
import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image

from ..util import HWC3, resize_image, safe_step
from .ted import TED


class TEEDdetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, subfolder=None):
        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(
                pretrained_model_or_path, filename, subfolder=subfolder
            )

        model = TED()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        safe_steps=2,
        output_type="pil",
    ):
        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
            output_type = output_type or "pil"
        else:
            output_type = output_type or "np"

        original_height, original_width, _ = input_image.shape

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
        height, width, _ = input_image.shape
        with torch.no_grad():
            image_teed = torch.from_numpy(input_image.copy()).float().to(device)
            image_teed = rearrange(image_teed, "h w c -> 1 c h w")
            edges = self.model(image_teed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [
                cv2.resize(e, (width, height), interpolation=cv2.INTER_LINEAR)
                for e in edges
            ]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe_steps != 0:
                edge = safe_step(edge, safe_steps)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge
        detected_map = HWC3(detected_map)

        detected_map = cv2.resize(
            detected_map,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
