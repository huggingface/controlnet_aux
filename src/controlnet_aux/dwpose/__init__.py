# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import torch
import numpy as np
from PIL import Image

from ..util import HWC3, resize_image
from . import util


def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    if draw_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if draw_hand:
        canvas = util.draw_handpose(canvas, hands)
    if draw_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas

class DWposeDetector:
    def __init__(self, det_config=None, det_ckpt=None, pose_config=None, pose_ckpt=None, device="cpu"):
        from .wholebody import Wholebody

        self.pose_estimation = Wholebody(det_config, det_ckpt, pose_config, pose_ckpt, device)
    
    def to(self, device):
        self.pose_estimation.to(device)
        return self
    
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, include_body=True, include_hand=True, include_face=True, hand_and_face=None, output_type="pil", **kwargs):
        if hand_and_face is not None:
            include_hand = hand_and_face
            include_face = hand_and_face

        input_image = cv2.cvtColor(np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            
            detected_map = draw_pose(pose, H, W, draw_body=include_body, draw_hand=include_hand, draw_face=include_face)
            detected_map = HWC3(detected_map)
            
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            if output_type == "pil":
                detected_map = Image.fromarray(detected_map)
                
            return detected_map
