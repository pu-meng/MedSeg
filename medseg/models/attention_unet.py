from monai.networks.nets import AttentionUnet

def build_attention_unet(in_channels=1, out_channels=2):
    return AttentionUnet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
    )