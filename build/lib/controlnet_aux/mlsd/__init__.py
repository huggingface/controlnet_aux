import cv2
import numpy as np
import torch
import os

from huggingface_hub import hf_hub_download
from .models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines
from PIL import Image
from ..open_pose.util import HWC3, resize_image


class MLSDdetector:
    def __init__(self, model):
        self.model = model.eval()

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None):
        filename = filename or "annotator/ckpts/mlsd_large_512_fp32.pth"
        model_path = hf_hub_download(pretrained_model_or_path, filename)

        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path), strict=True)

        return cls(model)

    def __call__(self, input_image, thr_v=0.1, thr_d=0.1, detect_resolution=512, image_resolution=512, return_pil=True):
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d)
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        except Exception as e:
            pass

        detected_map = img_output[:, :, 0]

        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        if return_pil:
            detected_map = Image.fromarray(detected_map)

        return detected_map
