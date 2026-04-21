"""
transforms_offline.py
=====================
适配离线预处理后的 .pt 文件.

在线模式:每次加载代价搞,num_samples=4
离线模式：预处理好，直接切 patch，num_samples=1,repeats=3,速度大幅提升.

验证时不做任何增强(离线已处理好,直接用).
"""

from monai.transforms.croppad.dictionary import RandCropByLabelClassesd
from monai.transforms.spatial.dictionary import RandFlipd, RandZoomd, RandRotated, RandSimulateLowResolutiond, Rand3DElasticd
from monai.transforms.intensity.dictionary import (
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandScaleIntensityd,
)
from monai.transforms.utility.dictionary import EnsureTyped

from monai.transforms.croppad.dictionary import SpatialPadd

import warnings
import torch
from monai.transforms.compose import Compose

warnings.filterwarnings("ignore", message=".*no available indices of class.*")


def build_train_transforms(patch_size=(144, 144, 144), ratios=(0.0, 1.0)):
    """
    训练 transforms:只做随机增强.
    输入已经是归一化+裁剪好的 tensor,从 .pt 直接加载.
    """
    return Compose(
        [
            # ── patch 采样 ────────────────────────────────────────────────
            # 保证体积不小于 patch_size(有些裁剪后体积可能比patch小)
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            # 按类别比例采样 patch,保证 tumor 被采到
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                num_classes=2,  # 0=背景, 1=前景(肝脏+肿瘤)/或者0=肝脏,1=肿瘤
                num_samples=1,  # 每个病例切1个patch
                ratios=list(ratios),  # tumor占更大比例
                allow_smaller=True,
            ),
            # ── 几何增强 ──────────────────────────────────────────────────
            # Elastic deformation：对应nnUNet ElasticDeformationTransform, p=0.2
            # sigma控制形变平滑度,magnitude控制形变幅度
            Rand3DElasticd(
                keys=["image", "label"],
                sigma_range=(3.0, 5.0),
                magnitude_range=(100.0, 200.0),
                prob=0.2,
                mode=["bilinear", "nearest"],
                padding_mode="border",
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # 连续旋转：任意角度（nnUNet用±180°），比rotate90（只有4种）覆盖更多姿态
            # prob=0.2对应nnUNet的p_rotation=0.2
            RandRotated(
                keys=["image", "label"],
                range_x=3.14159/6.0,  # 3.14159弧度=180度
                range_y=3.14159/6.0,
                range_z=3.14159/6.0,
                prob=0.2,
                mode=["bilinear", "nearest"],
                padding_mode="border",
            ),
            # ── 尺度增强：zoom in帮助小肿瘤，zoom out帮助大肿瘤上下文 ──────
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.7,
                max_zoom=1.4,
                mode=["trilinear", "nearest"],
                prob=0.2,
                keep_size=True,
            ),
            # ── 强度增强(只作用于image)────────────────────────────────────
            # 亮度乘法：对应nnUNet MultiplicativeBrightnessTransform(multiplier_range=(0.75,1.25)), p=0.15
            #factors=(-0.25,0.25)是从这里面随机采样一个值f,然后image_out=(1+f)*image
            RandScaleIntensityd(keys=["image"], factors=(-0.25, 0.25), prob=0.15),
            # 高斯噪声：对应nnUNet GaussianNoiseTransform(noise_variance=(0,0.1)), p=0.1
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
            # 高斯模糊：对应nnUNet GaussianBlurTransform(blur_sigma=(0.5,1.0)), p=0.2
           #高斯平滑叫做高斯模糊,两者是同一种操作
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.0),#sigma_x是x方向的标准差,sigma_x=(0.5,1.0)表示随机取一个数作为标准差
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
                prob=0.2,
            ),
           #RandAdjustContrastd这个是先将图像归一化到[0,1]
           #然后逐像素做幂运算,pixel_out = pixel_in^gamma, gamma>1会让图像更暗,gamma<1会让图像更亮
           #再映射回原始值域,retain_stats=True表示保持原图的均值和方差不变,只改变对比度
           

            RandAdjustContrastd(
                keys=["image"], prob=0.15, gamma=(0.75, 1.25), retain_stats=True
            ),
            # 低分辨率模拟：先用低质量插值(如最近邻)将图像缩小到原本的zoom_range比例
            #然后用线性插值放大回原本的尺寸
            #结果:产生模糊/块状伪影,模拟低分辨率扫描
            #zoom_range=(0.5,1.0)先随机取一个值在0.5到1.0的之间;这里的zoom_range越小,模糊程度越大
            RandSimulateLowResolutiond(
                keys=["image"],
                zoom_range=(0.5, 1.0),
                prob=0.25,
            ),
            # Gamma变换第一次：先反转图像再做gamma，对应nnUNet GammaTransform(p_invert_image=1, p_retain_stats=1), p=0.1
            RandAdjustContrastd(
                keys=["image"],
                prob=0.1,
                gamma=(0.7, 1.5),
                invert_image=True,
                retain_stats=True,
            ),
            # Gamma变换第二次：正常gamma，对应nnUNet GammaTransform(p_invert_image=0, p_retain_stats=1), p=0.3
            RandAdjustContrastd(
                keys=["image"],
                prob=0.3,
                gamma=(0.7, 1.5),
                retain_stats=True,
            ),
            # ── 类型保证 ──────────────────────────────────────────────────
            EnsureTyped(
                keys=["image", "label"],
                dtype=(torch.float32, torch.int64),
            ),
        ]
    )


def build_val_transforms():
    """
    验证 transforms:离线已处理好,什么都不用做.
    直接返回 EnsureTyped 保证类型正确即可.
    """
    return Compose(
        [
            EnsureTyped(
                keys=["image", "label"],
                dtype=(torch.float32, torch.int64),
            ),
        ]
    )
