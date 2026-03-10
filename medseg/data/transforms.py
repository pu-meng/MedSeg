from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandScaleIntensityd,
    EnsureTyped,
)
from monai.transforms import SpatialPadd, ResizeWithPadOrCropd
import torch

from monai.transforms import RandCropByLabelClassesd

LIVER_WIN_MIN = -13.7
LIVER_WIN_MAX = 188.3


def build_train_transforms(patch_size=(96, 96, 96), ratios=(0, 0.05, 0.95)):
    """
    Stage1 目标:稳定训练,不追求花哨
    - Load + 通道 + 方向统一
    - spacing 统一(训练稳定)
    - intensity 归一化(CT 推荐 window)
    - foreground crop
    - pos/neg patch 采样(保证前景)
    - PyTorch = 通用深度学习引擎
    - MONAI    = 医学影像专用工具箱
    - pin_memory = 加载到 GPU 速度更快,一般是最好开着不亏;但内存小的话就关了
    - spacing = (0.8mm, 0.8mm, 5.0mm)
    - 像素是2D,体素是3D
    - Spacingd(pixdim=(1.5,1.5,1.5)) 把每个体素变成15mm的正方体
    - RAS = Right-Anterior-Superior,Anterior是前面,Superior是上面
    -
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.88, 0.88, 0.88),
                mode=("bilinear", "nearest"),
            ),
            
            ScaleIntensityRanged(
                keys=["image"],
                a_min=LIVER_WIN_MIN,
                a_max=LIVER_WIN_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # 从RandCropByPosNegLabeld中到RnadRotate90d 都是只有训练数据集需要的数据增强
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                num_classes=3,  # 0/1/2
                num_samples=4,  # 每个病例切2个patch
                ratios=list(ratios),  # ✅ 重点:tumor(2)占更大比例
                allow_smaller=True,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            # 任意角度旋转(nnUNet的核心增强)
            # range_x,range_y,range_z是旋转角度,prob是旋转概率,mode是插值方式,padding_mode是填充方式
            # 0.523弧度大约是29.6度
            RandRotated(
                keys=["image", "label"],
                range_x=0.523,
                range_y=0.523,
                range_z=0.523,
                prob=0.3,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            # 4.随机缩放(模拟器官大小变化)
            # 0.85-1.25倍是nnUNet的缩放范围
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.85,
                max_zoom=1.25,
                prob=0.3,
                mode=("bilinear", "nearest"),
                padding_mode="constant",
            ),
            # 5.高斯噪声,模拟扫描仪噪声
            RandGaussianNoised(
                keys=["image"],
                prob=0.15,
                mean=0.0,
                std=0.1,
            ),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
                prob=0.15,
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.15,
                gamma=(0.5, 1.5),
            ),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.15),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
        ]
    )


def build_val_transforms():
    """
    验证不做随机增强,只做确定性预处理
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                   keys=["image", "label"],
                  pixdim=(0.88, 0.88, 0.88),
                 mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=LIVER_WIN_MIN,
                a_max=LIVER_WIN_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
        ]
    )



