import warnings
from typing import Union

import cv2
import numpy as np
from PIL import Image

from ..util import HWC3, resize_image
from .mediapipe_face_common import generate_annotation


class MediapipeFaceDetector:
    def __call__(self,
                 input_image: Union[np.ndarray, Image.Image] = None,
                 max_faces: int = 1,
                 min_confidence: float = 0.5,
                 return_pil: bool = None,
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 output_type: str = "pil",
                 **kwargs):

        if "image" in kwargs:
            warnings.warn("image is deprecated, please use `input_image=...` instead.", DeprecationWarning)
            input_image = kwargs.pop("image")
        
        if input_image is None:
            raise ValueError("input_image must be defined.")

        if return_pil is not None:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if return_pil else "np"

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        detected_map = generate_annotation(input_image, max_faces, min_confidence)
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
