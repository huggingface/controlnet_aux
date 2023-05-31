"""
This file contains a Processor that can be used to process images with controlnet aux processors
"""
import io
import logging
from typing import Union, Dict, Optional

from PIL import Image

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

CHOICES = "Choices for the processor are hed, midas, mlsd, openpose, pidinet, normalbae, lineart, lineart_coarse, lineart_anime, canny, content_shuffle, zoe, mediapipe_face"

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

LOGGER = logging.getLogger(__name__)


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
    def __init__(self, processor_id: str, params: Optional[Dict] = None) -> None:
        """Processor that can be used to process images with controlnet aux processors

        Args:
            processor_id (str): processor name, options are 'hed, midas, mlsd, openpose,
                                pidinet, normalbae, lineart, lineart_coarse, lineart_anime,
                                canny, content_shuffle, zoe, mediapipe_face
            params (Optional[Dict]): parameters for the processor
        """
        LOGGER.info("Loading %s".format(processor_id))
        self.processor_id = processor_id
        self.processor = self.load_processor(self.processor_id)

        # load default params
        self.params = MODEL_PARAMS[self.processor_id]
        # update with user params
        if params:
            self.params.update(params)

        self.resize = self.params.pop('resize', False)
        if self.resize:
            LOGGER.warning(f"Warning: {self.processor_id} will resize image to {self.resize}x{self.resize}")

    def load_processor(self, processor_id: str) -> 'Processor':
        """Load controlnet aux processors

        Args:
            processor_id (str): processor name

        Returns:
            Processor: controlnet aux processor
        """
        if processor_id not in MODELS:
            raise ValueError(f"Processor {processor_id} not found. {CHOICES}")

        processor = MODELS[processor_id]['class']

        # check if the proecssor is a checkpoint model
        if MODELS[processor_id]['checkpoint']:
            processor = processor.from_pretrained("lllyasviel/Annotators")
        else:
            processor = processor()
        return processor

    def __call__(self, image: Union[Image.Image, bytes],
                 to_pil: bool = True) -> Union[Image.Image, bytes]:
        """processes an image with a controlnet aux processor

        Args:
            image (Union[Image.Image, bytes]): input image in bytes or PIL Image
            to_pil (bool): whether to return bytes or PIL Image

        Returns:
            Union[Image.Image, bytes]: processed image in bytes or PIL Image
        """
        # check if bytes or PIL Image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        if self.resize:
            image = image.resize((self.resize, self.resize))

        processed_image = self.processor(image, **self.params)

        if to_pil:
            return processed_image
        else:
            output_bytes = io.BytesIO()
            processed_image.save(output_bytes, format='JPEG')
            return output_bytes.getvalue()
