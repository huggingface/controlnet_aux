# Code based based from the repository comfyui_controlnet_aux:
# https://github.com/Fannovel16/comfyui_controlnet_aux/blob/main/src/controlnet_aux/lineart_standard/__init__.py
import cv2
import numpy as np
from PIL import Image

from ..util import HWC3, resize_image


class LineartStandardDetector:
    def __call__(
        self,
        input_image=None,
        guassian_sigma=6.0,
        intensity_threshold=8,
        detect_resolution=512,
        output_type="pil",
    ):
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        else:
            output_type = output_type or "np"

        original_height, original_width, _ = input_image.shape

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        x = input_image.astype(np.float32)
        g = cv2.GaussianBlur(x, (0, 0), guassian_sigma)
        intensity = np.min(g - x, axis=2).clip(0, 255)
        intensity /= max(16, np.median(intensity[intensity > intensity_threshold]))
        intensity *= 127
        detected_map = intensity.clip(0, 255).astype(np.uint8)

        detected_map = HWC3(detected_map)

        detected_map = cv2.resize(
            detected_map,
            (original_width, original_height),
            interpolation=cv2.INTER_CUBIC,
        )

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
