from medseg.models.unet3d import build_unet3d
from medseg.models.unetr import build_unetr
from medseg.models.attention_unet import build_attention_unet
from medseg.models.segresnet import build_segresnet
from medseg.models.dynunet import build_dynunet


def build_model(name, in_channels=1, out_channels=2, img_size=(96, 96, 96)):
    name = name.lower()
    if name in ["unet", "unet3d"]:
        return build_unet3d(in_channels, out_channels)
    if name in ["unetr"]:
        return build_unetr(in_channels, out_channels, img_size=img_size)
    if name in ["attention_unet", "attunet"]:
        return build_attention_unet(in_channels, out_channels)
    if name in ["segresnet"]:
        return build_segresnet(in_channels, out_channels)
    if name in ["dynunet", "nnunet"]:
        return build_dynunet(in_channels, out_channels)
    raise ValueError(f"Unknown model: {name}")
