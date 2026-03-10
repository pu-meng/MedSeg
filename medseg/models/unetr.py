from monai.networks.nets import UNETR
import inspect


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