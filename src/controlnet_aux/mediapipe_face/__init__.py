from typing import Union

import cv2
import numpy as np
from PIL import Image

from ..util import HWC3, resize_image
from .mediapipe_face_common import generate_annotation


class MediapipeFaceDetector:
    def __call__(self,
                 input_image: Union[np.ndarray, Image.Image],
                 max_faces: int = 1,
                 min_confidence: float = 0.5,
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 return_pil: bool = True):
        
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        detected_map = generate_annotation(input_image, max_faces, min_confidence)
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        if return_pil is True:
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
