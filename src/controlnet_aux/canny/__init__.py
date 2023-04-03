import cv2
import numpy as np
from PIL import Image
from ..util import HWC3

class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        
        input_type = "np"
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        img = HWC3(img)
        img = cv2.Canny(img, low_threshold, high_threshold)
        
        img = Image.fromarray(img)
        img = img.convert("RGB")
            
        return img
