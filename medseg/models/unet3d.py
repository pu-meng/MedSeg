from monai.networks.nets.unet import UNet


def build_unet3d(in_channels=1, out_channels=2):
    """
    - channels=(32, 64, 128, 256, 320) 是每一层encoder的特征通道数
    - strides=(2, 2, 2, 2) 是每次下采样的倍率
    - num_res_units=2 是每个encoder block中的残差单元数
    """
    # Task02_Heart: 背景+心脏 => 2 类
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
