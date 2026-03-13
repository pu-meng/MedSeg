from monai.networks.nets import UNETR
import inspect
"""
UNETR:用transformer做encoder,CNN做decoder
Image
 ↓
Patch Embedding
 ↓
Transformer Encoder
 ↓      ↓      ↓      ↓
Skip1  Skip2  Skip3  Skip4
  │      │      │      │
  └──────Decoder (CNN)──────→ segmentation

  UNETR:使用ViT Transformer
Swin Unetr:使用Swin Transformer,window attention

论文中最常用的是nnUnet/DynUnet,因为它的性能好,而且实现简单,不需要复杂的Transformer结构和训练技巧
SwinUnetr:性能提升有限,而且实现复杂,需要调整很多超参数,不太适合初学者
nnUnet v2;
Transformer在小数据上容易过拟合,CNN训练稳定,超参数简单


"""

def build_unetr(in_channels=1, out_channels=2, img_size=(96, 96, 96)):
    """
    -perceptron:感知机
    -pos_embed:位置嵌入
    -    kwargs["pos_embed"] = "perceptron" 这个含义是用一个小的全连接网络学习位置编码
    - 类似def f(a,b,c=1) 函数签名=函数的参数列表结构
    - UNETR=tranformer+unet

    """
    kwargs = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )

    sig = inspect.signature(UNETR.__init__)
    if "pos_embed" in sig.parameters:
        kwargs["pos_embed"] = "perceptron"

    return UNETR(**kwargs)
# sig = inspect.signature(UNETR.__init__)
# print(sig.parameters)