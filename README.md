# ControlNet auxiliary models

This is a PyPi installable package of [lllyasviel's ControlNet Annotators](https://github.com/lllyasviel/ControlNet/tree/main/annotator)

The code is copy-pasted from the respective folders in https://github.com/lllyasviel/ControlNet/tree/main/annotator and connected to [the 🤗 Hub](https://huggingface.co/lllyasviel/Annotators).

All credit & copyright goes to https://github.com/lllyasviel .

## Install

```
pip install controlnet-aux==0.0.7
```

To support DWPose which is dependent on MMDetection, MMCV and MMPose
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```
## Usage


You can use the processor class, which can load each of the auxiliary models with the following code
```python
import requests
from PIL import Image
from io import BytesIO

from controlnet_aux.processor import Processor

# load image
url = "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"

response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

# load processor from processor_id
# options are:
# ["canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
#  "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
#  "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
#  "scribble_hed, "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
#  "softedge_pidinet", "softedge_pidsafe", "dwpose"]
processor_id = 'scribble_hed'
processor = Processor(processor_id)

processed_image = processor(img, to_pil=True)
```

Each model can be loaded individually by importing and instantiating them as follows
```python
from PIL import Image
import requests
from io import BytesIO
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector

# load image
url = "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"

response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

# load checkpoints
hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
mobile_sam = SamDetector.from_pretrained("dhkim2810/MobileSAM", model_type="vit_t", filename="mobile_sam.pt")
leres = LeresDetector.from_pretrained("lllyasviel/Annotators")

# specify configs, ckpts and device, or it will be downloaded automatically and use cpu by default
# det_config: ./src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py
# det_ckpt: https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
# pose_config: ./src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py
# pose_ckpt: https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dwpose = DWposeDetector(det_config=det_config, det_ckpt=det_ckpt, pose_config=pose_config, pose_ckpt=pose_ckpt, device=device)

# instantiate
canny = CannyDetector()
content = ContentShuffleDetector()
face_detector = MediapipeFaceDetector()


# process
processed_image_hed = hed(img)
processed_image_midas = midas(img)
processed_image_mlsd = mlsd(img)
processed_image_open_pose = open_pose(img, hand_and_face=True)
processed_image_pidi = pidi(img, safe=True)
processed_image_normal_bae = normal_bae(img)
processed_image_lineart = lineart(img, coarse=True)
processed_image_lineart_anime = lineart_anime(img)
processed_image_zoe = zoe(img)
processed_image_sam = sam(img)
processed_image_leres = leres(img)

processed_image_canny = canny(img)
processed_image_content = content(img)
processed_image_mediapipe_face = face_detector(img)
processed_image_dwpose = dwpose(img)
```

### Image resolution

In order to maintain the image aspect ratio, `detect_resolution`, `image_resolution` and calls to `resize_image` need to be using multiple of `32`.  
Otherwise images will be resized to work correctly.  
Resolution can be set to `None` to prevent resize. This may lead to RunTimeError.  
