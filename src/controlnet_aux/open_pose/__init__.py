import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import cv2
import numpy as np
from PIL import Image
from .util import HWC3, resize_image, draw_bodypose, draw_handpose, draw_facepose
from .body import Body
from .hand import handDetect, Hand
from .face import faceDetect, Face
from huggingface_hub import hf_hub_download
# from .hand import Hand
# from annotator.util import annotator_ckpts_path


#body_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth"
#hand_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth"


def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = draw_bodypose(canvas, candidate, subset)

    if draw_hand:
        canvas = draw_handpose(canvas, hands)

    if draw_face:
        canvas = draw_facepose(canvas, faces)

    return canvas


class OpenposeDetector:
    def __init__(self, body_estimation, hand_estimation=None, face_estimation=None):
        self.body_estimation = body_estimation
        self.hand_estimation = hand_estimation
        self.face_estimation = face_estimation

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, hand_filename=None, face_filename=None, cache_dir=None):

        if pretrained_model_or_path == "lllyasviel/ControlNet":
            filename = filename or "annotator/ckpts/body_pose_model.pth"
            hand_filename = hand_filename or "annotator/ckpts/hand_pose_model.pth"
            face_filename = face_filename or "facenet.pth"

            face_pretrained_model_or_path = "lllyasviel/Annotators"
        else:
            filename = filename or "body_pose_model.pth"
            hand_filename = hand_filename or "hand_pose_model.pth"
            face_filename = face_filename or "facenet.pth"

            face_pretrained_model_or_path = pretrained_model_or_path

        if os.path.isdir(pretrained_model_or_path):
            body_model_path = os.path.join(pretrained_model_or_path, filename)
            hand_model_path = os.path.join(pretrained_model_or_path, hand_filename)
            face_model_path = os.path.join(face_pretrained_model_or_path, face_filename)
        else:
            body_model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)
            hand_model_path = hf_hub_download(pretrained_model_or_path, hand_filename, cache_dir=cache_dir)
            face_model_path = hf_hub_download(face_pretrained_model_or_path, face_filename, cache_dir=cache_dir)

        body_estimation = Body(body_model_path)
        hand_estimation = Hand(hand_model_path)
        face_estimation = Face(face_model_path)

        return cls(body_estimation, hand_estimation, face_estimation)

    def __call__(self, input_image, detect_resolution=512, image_resolution=512, hand_and_face=False, return_pil=True):
        # hand = False
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        input_image = input_image[:, :, ::-1].copy()
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(input_image)
            hands = []
            faces = []
            if hand_and_face:
                # Hand
                hands_list = handDetect(candidate, subset, input_image)
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(input_image[y:y+w, x:x+w, :]).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                        hands.append(peaks.tolist())
                # Face
                faces_list = faceDetect(candidate, subset, input_image)
                for x, y, w in faces_list:
                    heatmaps = self.face_estimation(input_image[y:y+w, x:x+w, :])
                    peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                        faces.append(peaks.tolist())

            if candidate.ndim == 2 and candidate.shape[1] == 4:
                candidate = candidate[:, :2]
                candidate[:, 0] /= float(W)
                candidate[:, 1] /= float(H)

            bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            canvas = draw_pose(pose, H, W)

        detected_map = HWC3(canvas)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        if return_pil:
            detected_map = Image.fromarray(detected_map)

        return detected_map
