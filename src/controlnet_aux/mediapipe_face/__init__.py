from typing import Union

from .mediapipe_face_common import generate_annotation
from PIL import Image
import numpy as np


class MediapipeFaceDetector:
    def __call__(self,
                 image: Union[np.ndarray, Image.Image],
                 max_faces: int = 1,
                 min_confidence: float = 0.5,
                 return_pil: bool = True):
        
        if isinstance(image, Image.Image) is True:
            image = np.array(image)
        
        face = generate_annotation(image, max_faces, min_confidence)
        
        if return_pil is True:
            face = Image.fromarray(face)
            
        return face