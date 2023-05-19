# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from PIL import Image
import numpy as np
import torch
import gc

from huggingface_hub import hf_hub_download

from .build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
from .predictor import SamPredictor
from .automatic_mask_generator import SamAutomaticMaskGenerator


class SamDetector:
    def __init__(self, mask_generator: SamAutomaticMaskGenerator):
        self.mask_generator = mask_generator
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, model_type="vit_h", filename="sam_vit_h_4b8939.pth", cache_dir=None):
        """
        Possible model_type : vit_h, vit_l, vit_b
        download weights from https://github.com/facebookresearch/segment-anything
        """
        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)  
        
        sam = sam_model_registry[model_type](checkpoint=model_path)
        
        if torch.cuda.is_available():
            sam.to("cuda")
        
        mask_generator = SamAutomaticMaskGenerator(sam)

        return cls(mask_generator)


    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        h, w =  anns[0]['segmentation'].shape
        final_img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.empty((m.shape[0], m.shape[1], 3), dtype=np.uint8)
            for i in range(3):
                img[:,:,i] = np.random.randint(255, dtype=np.uint8)
            final_img.paste(Image.fromarray(img, mode="RGB"), (0, 0), Image.fromarray(np.uint8(m*255)))
        
        return final_img

    def __call__(self, image: Image.Image) -> Image.Image:
        # Generate Masks
        if isinstance(image, Image.Image):
            image = np.array(image)

        masks = self.mask_generator.generate(image)
        torch.cuda.empty_cache()
        # Create map
        map = self.show_anns(masks)
        del masks
        gc.collect()
        return map