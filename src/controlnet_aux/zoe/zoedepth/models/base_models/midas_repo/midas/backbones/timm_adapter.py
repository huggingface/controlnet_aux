import importlib.metadata
import timm

# Check the installed timm version
timm_version = importlib.metadata.version("timm")
is_new_timm = timm_version >= "1.0.0"

def create_model_adapter(model_name, pretrained=False, **kwargs):
    """
    Adapter function for creating models with timm that works with both old (0.6.7) and new (1.0+) versions.
    """
    if is_new_timm:
        # In timm 1.0+, 'pretrained' is deprecated in favor of 'pretrained_cfg' or explicit pretrained_cfg_url
        if pretrained:
            return timm.create_model(model_name, pretrained_cfg='default', **kwargs)
        else:
            return timm.create_model(model_name, pretrained_cfg=None, **kwargs)
    else:
        # Old timm behavior
        return timm.create_model(model_name, pretrained=pretrained, **kwargs) 