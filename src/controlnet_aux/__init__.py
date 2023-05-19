__version__ = "0.0.3"

from .hed import HEDdetector
from .midas import MidasDetector
from .mlsd import MLSDdetector
from .open_pose import OpenposeDetector
from .pidi import PidiNetDetector
from .normalbae import NormalBaeDetector
from .lineart import LineartDetector
from .lineart_anime import LineartAnimeDetector
from .zoe import ZoeDetector

from .canny import CannyDetector
from .shuffle import ContentShuffleDetector
from .mediapipe_face import MediapipeFaceDetector
from .segment_anything import SAM