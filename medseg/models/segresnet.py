from monai.networks.nets import SegResNet

def build_segresnet(in_channels=1, out_channels=2):
    return SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=32,
    )