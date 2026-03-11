from monai.networks.nets import DynUNet

def build_dynunet(in_channels=1, out_channels=2):
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
        strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        upsample_kernel_size=[[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        deep_supervision=False,
    )