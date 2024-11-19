import cv2
import numpy as np
from easy_dwpose import DWposeDetector as Detector
from PIL import Image

from ..util import resize_image
from . import util


def draw_pose(
    pose, height: int, width: int, include_face: bool = True, include_hands: bool = True, *args, **kwargs
) -> np.ndarray:
    canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    candidate = pose["bodies"]
    subset = pose["body_scores"]
    canvas = util.draw_bodypose(canvas, candidate, subset)

    if include_face:
        faces = pose["faces"]
        canvas = util.draw_facepose(canvas, faces)

    if include_hands:
        hands = pose["hands"]
        canvas = util.draw_handpose(canvas, hands)

    return canvas


class DWposeDetector:
    def __init__(self, device="cpu"):
        self.pose_estimation = Detector(device=device)

    def __call__(self, input_image: Image, image_resolution: int = 512, output_type: str = "pil", **kwargs):
        detected_map = self.pose_estimation(input_image, output_type=output_type, draw_pose=draw_pose, **kwargs)

        input_image = np.array(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        if output_type == "pil":
            detected_map = detected_map.resize((W, H), Image.BILINEAR)
        else:
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        return detected_map
