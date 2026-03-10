from medseg.models.unet3d import build_unet3d
from medseg.models.unetr import build_unetr


def build_model(name: str, in_channels=1, out_channels=2, img_size=(96, 96, 96)):
    name = name.lower()
    if name in ["unet", "unet3d"]:
        return build_unet3d(in_channels, out_channels)
    if name in ["unetr"]:
        return build_unetr(in_channels, out_channels, img_size=img_size)
    raise ValueError(f"Unknown model: {name}")
