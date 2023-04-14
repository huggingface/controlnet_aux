import cv2
import numpy as np
from PIL import Image
from ..util import HWC3

class CannyDetector:
    def __call__(self, img, low_threshold=100, high_threshold=200):
        
        input_type = "np"
        if isinstance(img, Image.Image):
            img = np.array(img)
            input_type = "pil"
        
        img = HWC3(img)
        img = cv2.Canny(img, low_threshold, high_threshold)
        
        if input_type == "pil":
            img = Image.fromarray(img)
            img = img.convert("RGB")
            
        return img
