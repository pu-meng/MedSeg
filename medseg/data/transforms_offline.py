"""
transforms_offline.py
=====================
适配离线预处理后的 .pt 文件.

离线已做(不需要再做):
  ✅ LoadImaged
  ✅ EnsureChannelFirstd
  ✅ ScaleIntensityRanged(归一化)
  ✅ CropForegroundd(前景裁剪)

训练时只做随机增强(在线):
  ✅ SpatialPadd
  ✅ RandCropByLabelClassesd
  ✅ RandFlipd
  ✅ RandRotate90d
  ✅ RandRotated
  ✅ RandZoomd
  ✅ RandGaussianNoised
  ✅ RandGaussianSmoothd
  ✅ RandAdjustContrastd
  ✅ RandScaleIntensityd

验证时不做任何增强(离线已处理好,直接用).
"""

import torch
from monai.transforms import (
    Compose,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandScaleIntensityd,
    EnsureTyped,
)
from monai.transforms import SpatialPadd


def build_train_transforms(patch_size=(144, 144, 144), ratios=(0.0, 0.05, 0.95)):
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
                num_classes=3,  # 0=bg, 1=liver, 2=tumor
                num_samples=1,  # 每个病例切4个patch
                ratios=list(ratios),  # tumor占更大比例
                allow_smaller=True,
            ),
            # ── 几何增强 ──────────────────────────────────────────────────
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            # 任意角度旋转(nnUNet核心增强)
            # RandRotated(
            #     keys=["image", "label"],
            #     range_x=0.523,  # ≈30度
            #     range_y=0.523,
            #     range_z=0.523,
            #     prob=0.3,
            #     mode=("bilinear", "nearest"),
            #     padding_mode="zeros",
            # ),
            # 随机缩放(模拟器官大小变化)
            # RandZoomd(
            #     keys=["image", "label"],
            #     min_zoom=0.85,
            #     max_zoom=1.25,
            #     prob=0.3,
            #     mode=("bilinear", "nearest"),
            #     padding_mode="constant",
            # ),
            # ── 强度增强(只作用于image)────────────────────────────────
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
            # RandGaussianSmoothd(
            #     keys=["image"],
            #     sigma_x=(0.5, 1.5),
            #     sigma_y=(0.5, 1.5),
            #     sigma_z=(0.5, 1.5),
            #     prob=0.15,
            # ),
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
