import numpy as np
import util
from easy_dwpose import DWposeDetector as Detector


def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]

    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    canvas = util.draw_handpose(canvas, hands)
    canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self, device="cpu"):
        self.pose_estimation = Detector(device=device)

    def __call__(self, input_image, **kwargs):
        detected_map = self.pose_estimation(input_image, draw_pose=draw_pose, **kwargs)
        return detected_map
