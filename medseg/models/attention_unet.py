from monai.networks.nets import AttentionUnet
"""
AttentionUnet:
Encoder feature
       │
       ▼
  Attention Gate
       │
       ▼
     concat
       │
       ▼
Decoder feature
Attentio Gate:
重要区域 → 保留
不重要区域 → 抑制

Attention Gate:输入两个信息
Encoder feature:来自编码器的特征图,包含了图像的空间信息和语义信息
Decoder feature:来自解码器的特征图,包含了当前解码器层
然后Attention Gate会产生一个权重图:
alpha(x) ∈ [0,1]
最终输出:
Attention output = alpha(x) * Encoder feature
alpha=0,完全抑制
alpha=1,完全保留


"""
def build_attention_unet(in_channels=1, out_channels=2):
    """
    spatial_dims=3,表示这是一个3D网络,
    输入:[B,C,D,H,W];
    channels=(32, 64, 128, 256, 320)
    表示每个stage的特征通道数
    strides=(2, 2, 2, 2)
    表示每个stage的卷积核的步长,也就是下采样的倍数
    Attention Unet提高有限
    """
    
    return AttentionUnet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
    )