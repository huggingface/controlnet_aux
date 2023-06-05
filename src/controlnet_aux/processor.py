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
                            MediapipeFaceDetector,
                            TilingDetector,
                            )

LOGGER = logging.getLogger(__name__)


MODELS = {
    # checkpoint models
    'scribble_hed': {'class': HEDdetector, 'checkpoint': True},
    'softedge_hed': {'class': HEDdetector, 'checkpoint': True},
    'scribble_hedsafe': {'class': HEDdetector, 'checkpoint': True},
    'softedge_hedsafe': {'class': HEDdetector, 'checkpoint': True},
    'depth_midas': {'class': MidasDetector, 'checkpoint': True},
    'mlsd': {'class': MLSDdetector, 'checkpoint': True},
    'openpose': {'class': OpenposeDetector, 'checkpoint': True},
    'openpose_face': {'class': OpenposeDetector, 'checkpoint': True},
    'openpose_faceonly': {'class': OpenposeDetector, 'checkpoint': True},
    'openpose_full': {'class': OpenposeDetector, 'checkpoint': True},
    'openpose_hand': {'class': OpenposeDetector, 'checkpoint': True},
    'scribble_pidinet': {'class': PidiNetDetector, 'checkpoint': True},
    'softedge_pidinet': {'class': PidiNetDetector, 'checkpoint': True},
    'scribble_pidsafe': {'class': PidiNetDetector, 'checkpoint': True},
    'softedge_pidsafe': {'class': PidiNetDetector, 'checkpoint': True},
    'normal_bae': {'class': NormalBaeDetector, 'checkpoint': True},
    'lineart_coarse': {'class': LineartDetector, 'checkpoint': True},
    'lineart_realistic': {'class': LineartDetector, 'checkpoint': True},
    'lineart_anime': {'class': LineartAnimeDetector, 'checkpoint': True},
    'depth_zoe': {'class': ZoeDetector, 'checkpoint': True},
    # instantiate
    'shuffle': {'class': ContentShuffleDetector, 'checkpoint': False},
    'mediapipe_face': {'class': MediapipeFaceDetector, 'checkpoint': False},
    'canny': {'class': CannyDetector, 'checkpoint': False},
    'tiling': {'class': TilingDetector, 'checkpoint': False},
}


MODEL_PARAMS = {
    'scribble_hed': {'resize': False, 'scribble': True},
    'softedge_hed': {'resize': False, 'scribble': False},
    'scribble_hedsafe': {'resize': False, 'scribble': True, 'safe': True},
    'softedge_hedsafe': {'resize': False, 'scribble': False, 'safe': True},
    'depth_midas': {'resize': 512},
    'mlsd': {'resize': False},
    'openpose': {'resize': False, 'include_body': True, 'include_hand': False, 'include_face': False},
    'openpose_face': {'resize': False, 'include_body': True, 'include_hand': False, 'include_face': True},
    'openpose_faceonly': {'resize': False, 'include_body': False, 'include_hand': False, 'include_face': True},
    'openpose_full': {'resize': False, 'include_body': True, 'include_hand': True, 'include_face': True},
    'openpose_hand': {'resize': False, 'include_body': False, 'include_hand': True, 'include_face': False},
    'scribble_pidinet': {'resize': False, 'safe': False, 'scribble': True},
    'softedge_pidinet': {'resize': False, 'safe': False, 'scribble': False},
    'scribble_pidsafe': {'resize': False, 'safe': True, 'scribble': True},
    'softedge_pidsafe': {'resize': False, 'safe': True, 'scribble': False},
    'normal_bae': {'resize': False},
    'lineart_realistic': {'resize': False, 'coarse': False},
    'lineart_coarse': {'resize': False, 'coarse': True},
    'lineart_anime': {'resize': False},
    'canny': {'resize': False},
    'shuffle': {'resize': False},
    'depth_zoe': {'resize': False},
    'mediapipe_face': {'resize': False},
    'tiling': {'resize': False},
}

CHOICES = f"Choices for the processor are {list(MODELS.keys())}"


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

        if processor_id not in MODELS:
            raise ValueError(f"{processor_id} is not a valid processor id. Please make sure to choose one of {', '.join(MODELS.keys())}".)

        self.processor_id = processor_id
        self.processor = self.load_processor(self.processor_id)

        # load default params
        self.params = MODELS[self.processor_id]
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
