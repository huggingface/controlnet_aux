import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import cv2
import numpy as np
from PIL import Image
from .util import HWC3, resize_image, draw_bodypose
from .body import Body
from huggingface_hub import hf_hub_download
# from .hand import Hand
# from annotator.util import annotator_ckpts_path


#body_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth"
#hand_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth"


class OpenposeDetector:
    def __init__(self, body_estimation):
        # body_modelpath = os.path.join(annotator_ckpts_path, "body_pose_model.pth")
        # hand_modelpath = os.path.join(annotator_ckpts_path, "hand_pose_model.pth")

#        if not os.path.exists(hand_modelpath):
#            from basicsr.utils.download_util import load_file_from_url
#            load_file_from_url(body_model_path, model_dir=annotator_ckpts_path)
            # load_file_from_url(hand_model_path, model_dir=annotator_ckpts_path)

        # self.body_estimation = Body(body_modelpath)
        # self.hand_estimation = Hand(hand_modelpath)
        self.body_estimation = body_estimation

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None):
        filename = filename or "annotator/ckpts/body_pose_model.pth"
        body_model_path = hf_hub_download(pretrained_model_or_path, filename)

        body_estimation = Body(body_model_path)

        return cls(body_estimation)

    def __call__(self, input_image, detect_resolution=512, image_resolution=512, return_pil=True):
        # hand = False
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        input_image = input_image[:, :, ::-1].copy()
        with torch.no_grad():
            candidate, subset = self.body_estimation(input_image)
            canvas = np.zeros_like(input_image)
            canvas = draw_bodypose(canvas, candidate, subset)
#            if hand:
#                hands_list = util.handDetect(candidate, subset, oriImg)
#                all_hand_peaks = []
#                for x, y, w, is_left in hands_list:
#                    peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :])
#                    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
#                    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
#                    all_hand_peaks.append(peaks)
#                canvas = util.draw_handpose(canvas, all_hand_peaks)
#        return canvas, dict(candidate=candidate.tolist(), subset=subset.tolist())
        detected_map = HWC3(canvas)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        if return_pil:
            detected_map = Image.fromarray(detected_map)

        return detected_map
