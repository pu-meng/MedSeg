"""
transforms_offline.py
=====================
适配离线预处理后的 .pt 文件.

在线模式：每次加载代价搞，num_samples=4
离线模式：预处理好，直接切 patch，num_samples=1，，repeats=3,速度大幅提升.

验证时不做任何增强(离线已处理好,直接用).
"""

import torch
from monai.transforms.compose import Compose

from monai.transforms.croppad.dictionary import RandCropByLabelClassesd
from monai.transforms.spatial.dictionary import RandFlipd, RandRotate90d
from monai.transforms.intensity.dictionary import (
    RandGaussianNoised,
    RandAdjustContrastd,
    RandScaleIntensityd,
)
from monai.transforms.utility.dictionary import EnsureTyped

from monai.transforms.croppad.dictionary import SpatialPadd


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
                num_samples=1,  # 每个病例切1个patch,这个要是增加会增加内存,repeats的增加可以起到同样的效果;
                ratios=list(ratios),  # tumor占更大比例
                allow_smaller=True,
            ),
            # ── 几何增强 ──────────────────────────────────────────────────
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
           
           
            # ── 强度增强(只作用于image)────────────────────────────────
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
          
            RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.5, 1.5)),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.15),
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
