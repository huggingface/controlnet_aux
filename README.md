# ControlNet auxiliary models

This is a PyPi installable package of [lllyasviel's ControlNet Annotators](https://github.com/lllyasviel/ControlNet/tree/main/annotator)

The code is copy-pasted from the respective folders in https://github.com/lllyasviel/ControlNet/tree/main/annotator and connected to [the 🤗 Hub](https://huggingface.co/lllyasviel/Annotators).

All credit & copyright goes to https://github.com/lllyasviel .

## Install

```
pip install controlnet-aux==0.0.2
```

## Usage

```python
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector, CannyDetector, MidasDetector

open_pose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")
midas = MidasDetector.from_pretrained("lllyasviel/ControlNet")
canny = CannyDetector()
```
