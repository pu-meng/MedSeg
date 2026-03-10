"""
Stage2 专用 transforms
输入: 已经裁剪好的 liver ROI (image + label)
label 语义: 0=背景(含肝脏实质), 1=肿瘤
"""

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandScaleIntensityd,
    EnsureTyped,
    CropForegroundd,
)

# 与 Stage1 完全一致的 HU 窗口 (nnUNet统计值)
WIN_MIN = -13.7
WIN_MAX = 188.3


def build_stage2_train_transforms(patch_size=(96, 96, 96), ratios=(0.1, 0.9)):
    """
    ratios: (背景, 肿瘤)
    Stage2 专注肿瘤, 所以肿瘤采样比例设高一点
    RandRotated 的旋转范围设置为 30 度 (0.523 弧度), 避免过大旋转导致肿瘤被裁剪掉
    RandZoom 的缩放范围设置为 0.85-1.25, 避免过大缩放导致肿瘤被裁剪掉或过度放大变形
    SpatialPadd的keys=["image", "label"] 可以保证 image 和 label 同时被 padding, 避免因为肿瘤靠近边界而被裁剪掉
    spatial_size=patch_size 可以保证 image 和 label 同时被裁剪到 patch_size 大小, 避免因为肿瘤靠近边界而被裁剪掉
    CropForegroundd 可以进一步裁剪掉大量背景, 避免因为肿瘤靠近边界而被裁剪掉
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
                a_min=WIN_MIN,
                a_max=WIN_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",  # 根据 label选择采样中心点
                spatial_size=patch_size,
                num_classes=2,  # 0=背景, 1=肿瘤
                num_samples=4,
                ratios=list(ratios),
                allow_smaller=True,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            RandRotated(
                keys=["image", "label"],
                range_x=0.523,
                range_y=0.523,
                range_z=0.523,
                prob=0.3,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.85,
                max_zoom=1.25,
                prob=0.3,
                mode=("bilinear", "nearest"),
                padding_mode="constant",
            ),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
                prob=0.15,
            ),
            RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.5, 1.5)),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.15),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
        ]
    )


def build_stage2_val_transforms():
    """验证不做随机增强, 只做确定性预处理"""
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
                a_min=WIN_MIN,
                a_max=WIN_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
        ]
    )
