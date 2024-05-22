# code based in https://github.com/TheMistoAI/ComfyUI-Anyline/blob/main/anyline.py
import os

import cv2
import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image
from skimage import morphology

from ..teed.ted import TED
from ..util import HWC3, resize_image, safe_step


class AnylineDetector:
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
        detect_resolution=1280,
        guassian_sigma=2.0,
        intensity_threshold=3,
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
            edge = safe_step(edge, 2)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        mteed_result = edge
        mteed_result = HWC3(mteed_result)

        x = input_image.astype(np.float32)
        g = cv2.GaussianBlur(x, (0, 0), guassian_sigma)
        intensity = np.min(g - x, axis=2).clip(0, 255)
        intensity /= max(16, np.median(intensity[intensity > intensity_threshold]))
        intensity *= 127
        lineart_result = intensity.clip(0, 255).astype(np.uint8)

        lineart_result = HWC3(lineart_result)

        lineart_result = self.get_intensity_mask(
            lineart_result, lower_bound=0, upper_bound=255
        )

        cleaned = morphology.remove_small_objects(
            lineart_result.astype(bool), min_size=36, connectivity=1
        )
        lineart_result = lineart_result * cleaned
        final_result = self.combine_layers(mteed_result, lineart_result)

        final_result = cv2.resize(
            final_result,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )

        if output_type == "pil":
            final_result = Image.fromarray(final_result)

        return final_result

    def get_intensity_mask(self, image_array, lower_bound, upper_bound):
        mask = image_array[:, :, 0]
        mask = np.where((mask >= lower_bound) & (mask <= upper_bound), mask, 0)
        mask = np.expand_dims(mask, 2).repeat(3, axis=2)
        return mask

    def combine_layers(self, base_layer, top_layer):
        mask = top_layer.astype(bool)
        temp = 1 - (1 - top_layer) * (1 - base_layer)
        result = base_layer * (~mask) + temp * mask
        return result
