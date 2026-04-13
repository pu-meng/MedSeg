from monai.networks.nets import DynUNet
"""
DynUnet=Dynamic U-Net
结构可以动态配置,非常适合3D医学图像
DynUnet=nnUnet使用的Unet结构版本
kernel_size=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
表示我的网络有5层,每层卷积核大小都是3x3x3
strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
表示第一层卷积步长是1,后面四层卷积步长都是2,也就是下采样
deep_supervision=False,表示不使用深监督,也就是不在每层输出一个分支来计算损失,只在最后一层输出分支来计算损失
普通的只有最后一层输出loss,
深监督是在每层输出一个分支来计算损失,然后加权求和
dynamic体现在可以自定义下采样次数,卷积核大小、步长、上采样卷积核大小等参数,非常灵活,适合不同尺寸的输入图像
U-Net:
Encoder(CNN),Decoder(CNN),Skip Connection
Transformer:
Encoder(Transformer),Decoder(Transformer),Skip Connection

"""
def build_dynunet(in_channels=1, out_channels=2, deep_supervision=True):
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
        strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        upsample_kernel_size=[[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        deep_supervision=deep_supervision,
    )


def build_dynunet_deep(in_channels=1, out_channels=2, deep_supervision=True):
    """6层更深版本，参数量~31M，对齐nnUNet Task003实际配置。"""
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
        strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        upsample_kernel_size=[[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        filters=[32,64,128,256,320,320],
        deep_supervision=deep_supervision,
    )