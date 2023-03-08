# ControlNet auxiliary models

This is a Pypi installable copied version of HED, MLSD and Human Pose auxiliary models: https://github.com/lllyasviel/ControlNet/tree/main/annotator

All credit goes to https://github.com/lllyasviel .
```
pip install controlnet-aux==0.0.1
```


```python
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector, CannyDetector, MidasDetector

open_pose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")
canny = CannyDetector()
midas = MidasDetector()
```
