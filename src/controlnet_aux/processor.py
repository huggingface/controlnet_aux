"""
This file contains a Processor that can be used to process images with controlnet aux processors
"""
import io
from typing import Union

from PIL import Image
import numpy as np
import torch
from controlnet_aux import (HEDdetector,
                            MidasDetector,
                            MLSDdetector,
                            OpenposeDetector,
                            PidiNetDetector,
                            NormalBaeDetector,
                            LineartDetector,
                            LineartAnimeDetector,
                            CannyDetector,
                            ContentShuffleDetector,
                            ZoeDetector,
                            MediapipeFaceDetector)


MODELS = {
    # checkpoint models
    'hed': {'class': HEDdetector, 'checkpoint': True},
    'midas': {'class': MidasDetector, 'checkpoint': True},
    'mlsd': {'class': MLSDdetector, 'checkpoint': True},
    'openpose': {'class': OpenposeDetector, 'checkpoint': True},
    'pidinet': {'class': PidiNetDetector, 'checkpoint': True},
    'normalbae': {'class': NormalBaeDetector, 'checkpoint': True},
    'lineart': {'class': LineartDetector, 'checkpoint': True},
    'lineart_coarse': {'class': LineartDetector, 'checkpoint': True},
    'lineart_anime': {'class': LineartAnimeDetector, 'checkpoint': True},
    'zoe': {'class': ZoeDetector, 'checkpoint': True}, 
    # instantiate
    'content_shuffle': {'class': ContentShuffleDetector, 'checkpoint': False},
    'mediapipe_face': {'class': MediapipeFaceDetector, 'checkpoint': False},
    'canny': {'class': CannyDetector, 'checkpoint': False},
}

# @patrickvonplaten, I can change this so people can pass their own parameters
# but for my use case I'm using this Dictionary
MODEL_PARAMS = {
    'hed': {'resize': False},
    'midas': {'resize': 512},
    'mlsd': {'resize': False},
    'openpose': {'resize': False, 'hand_and_face': True},
    'pidinet': {'resize': False, 'safe': True},
    'normalbae': {'resize': False},
    'lineart': {'resize': False, 'coarse': True},
    'lineart_coarse': {'resize': False, 'coarse': True},
    'lineart_anime': {'resize': False},
    'canny': {'resize': False},
    'content_shuffle': {'resize': False},
    'zoe': {'resize': False},
    'mediapipe_face': {'resize': False},
}


class Processor:
    def __init__(self, processor_id: str) -> 'Processor':
        """Processor that can be used to process images with controlnet aux processors

        Args:
            processor_id (str): processor name

        Returns:
            Processor: Processor object
        """
        print(f"Loading {processor_id} processor")
        self.processor_id = processor_id
        self.processor = self.load_processor(self.processor_id)
        self.params = MODEL_PARAMS[self.processor_id]
        self.resize = self.params.pop('resize', False)
        if self.resize:
            # print warning: image will be resized
            print(f"Warning: {self.processor_id} will resize image to {self.resize}x{self.resize}")

    def load_processor(self, processor_id: str):
        """Load controlnet aux processors

        Args:
            processor_id (str): processor name
        """
        processor = MODELS[processor_id]['class']

        if MODELS[processor_id]['checkpoint']:
            processor = processor.from_pretrained("lllyasviel/Annotators")
        else:
            processor = processor()
        return processor

    def __call__(self, image: Union[Image.Image, bytes],
                 to_bytes: bool = True) -> Union[Image.Image, bytes]:
        """processes an image with a controlnet aux processor

        Args:
            image (Union[Image.Image, bytes]): input image in bytes or PIL Image
            to_bytes (bool): whether to return bytes or PIL Image

        Returns:
            Union[Image.Image, bytes]: processed image in bytes or PIL Image
        """
        # check if bytes or PIL Image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        if self.resize:
            image = image.resize((self.resize, self.resize))

        processed_image = self.processor(image, **self.params)

        if to_bytes:
            output_bytes = io.BytesIO()
            processed_image.save(output_bytes, format='JPEG')
            return output_bytes.getvalue()
        else:
            return processed_image
