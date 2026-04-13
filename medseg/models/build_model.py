from medseg.models.unet3d import build_unet3d
from medseg.models.unetr import build_unetr
from medseg.models.attention_unet import build_attention_unet
from medseg.models.segresnet import build_segresnet
from medseg.models.dynunet import build_dynunet, build_dynunet_deep
from medseg.models.dynunet_ca import build_dynunet_ca
from medseg.models.swinunetr import build_swinunetr
"""
build_model.py
=================

统一模型构建接口 (Model Factory)。

该模块根据字符串名称构建不同的医学图像分割模型，
用于训练脚本中快速切换网络结构。

设计目的
--------
1. 统一模型入口，避免在 train.py 中写大量 if/else。
2. 方便实验对比，只需修改命令行参数 --model 即可切换模型。
3. 保持训练框架稳定，模型实现独立维护。

支持模型
--------
目前支持以下3D医学图像分割网络:

- UNet3D          : 经典3D U-Net卷积网络
- UNETR           : Transformer-based U-Net
- Attention UNet  : 带注意力门控的U-Net
- SegResNet       : ResNet结构的分割网络
- DynUNet         : nnUNet风格的动态U-Net

调用方式
--------
示例：

    model = build_model(
        name="dynunet",
        in_channels=1,
        out_channels=3,
        img_size=(96,96,96)
    )

返回：
    PyTorch nn.Module
"""


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
    if name in ["dynunet_deep", "nnunet_deep"]:
        return build_dynunet_deep(in_channels, out_channels)
    #这里的nnunet是dynunet的其他名字,本质是一样的
    
    if name in ["dynunet_ca", "nnunet_ca"]:
        return build_dynunet_ca(in_channels, out_channels)
    if name in ["swinunetr","swin_unetr"]:
        return build_swinunetr(in_channels, out_channels, img_size=img_size )
    raise ValueError(f"Unknown model: {name}")
