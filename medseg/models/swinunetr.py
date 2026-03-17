from monai.networks.nets.swin_unetr import SwinUNETR

"""
SwinUNETR=Swin transformer+UNETR
input (3D CT)
      │
patch partition
      │
Swin Transformer
      │
      │
      ├── feature1
      ├── feature2
      ├── feature3
      ├── feature4
      │
      ↓
   UNet decoder
      │
      ↓
 segmentation map
下面的参数feature size=Transformer embedding dim


"""


# def build_swinunetr(in_channels=1, out_channels=2, img_size=(96, 96, 96)):
#     return SwinUNETR(
#         img_size=img_size,
#         in_channels=in_channels,
#         out_channels=out_channels,
#         feature_size=24,
#         use_checkpoint=True,
#     )
def build_swinunetr(in_channels=1, out_channels=2, img_size=(96,96,96)):

    return SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=12,
        use_checkpoint=True,
    )