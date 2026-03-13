from monai.networks.nets import SegResNet
"""
SegResNet=医学分割版的ResNet+U-Net结构
init_filters=32,第一层卷积的通道数
32-64-128-256-512,通道数依次翻倍;
残差结构:
x → conv → conv
 \____________/
       +
      output
"""
def build_segresnet(in_channels=1, out_channels=2):
    return SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=32,
    )