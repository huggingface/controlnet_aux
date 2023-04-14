import os
import types
import torch
import numpy as np
from PIL import Image

from ..open_pose.util import HWC3, resize_image
from einops import rearrange
from .nets.NNET import NNET
from huggingface_hub import hf_hub_download
from .utils import utils
import torchvision.transforms as transforms


class NormalBaeDetector:
    def __init__(self, model):
        self.model = model.eval()

        if torch.cuda.is_available():
            self.model.cuda()

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None):
        filename = filename or "scannet.pt"
        model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)

        args = types.SimpleNamespace()
        args.mode = 'client'
        args.architecture = 'BN'
        args.pretrained = 'scannet'
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = utils.load_checkpoint(model_path, model)

        return cls(model)


    def __call__(self, input_image, detect_resolution=512, image_resolution=512, return_pil=True):
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
        image_normal = input_image
        with torch.no_grad():
            image_normal = torch.from_numpy(image_normal).float()
            model_device = next(iter(self.model.parameters())).device
            image_normal = image_normal.to(model_device)

            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)

            normal = self.model(image_normal)
            normal = normal[0][-1][:, :3]
            # d = torch.sum(normal ** 2.0, dim=1, keepdim=True) ** 0.5
            # d = torch.maximum(d, torch.ones_like(d) * 1e-5)
            # normal /= d
            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

            img = resize_image(normal_image, image_resolution)

            if return_pil:
                img = Image.fromarray(img)

            return img
