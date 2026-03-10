"""
calc_spacing.py
统计原始 Task03_Liver 数据集的 spacing 分布
输出:median spacing,用于决定预处理的 target spacing
用法:python calc_spacing.py --data_root /path/to/Task03_Liver
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
# /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        default="/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver",
    )
    args = p.parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.data_root, "imagesTr", "*.nii.gz")))
    assert len(img_paths) > 0, f"找不到数据: {args.data_root}"

    print(f"共 {len(img_paths)} 个 case,开始统计 spacing...")

    spacings = []
    sizes = []

    for img_p in tqdm(img_paths):
        nii = nib.load(img_p)
        shape = nii.shape[:3]  # (D, H, W)
        zooms = nii.header.get_zooms()[:3]  # (sz, sy, sx) 单位 mm
        spacings.append(zooms)
        sizes.append(shape)

    spacings = np.array(spacings)  # [N, 3]
    sizes = np.array(sizes)  # [N, 3]

    median_spacing = np.median(spacings, axis=0)
    min_spacing = spacings.min(axis=0)
    max_spacing = spacings.max(axis=0)

    median_size = np.median(sizes, axis=0)
    min_size = sizes.min(axis=0)
    max_size = sizes.max(axis=0)

    print("\n========== Spacing 统计(单位 mm)==========")
    print(
        f"median spacing : {median_spacing[0]:.3f}  {median_spacing[1]:.3f}  {median_spacing[2]:.3f}"
    )
    print(
        f"min    spacing : {min_spacing[0]:.3f}  {min_spacing[1]:.3f}  {min_spacing[2]:.3f}"
    )
    print(
        f"max    spacing : {max_spacing[0]:.3f}  {max_spacing[1]:.3f}  {max_spacing[2]:.3f}"
    )

    print("\n========== 原始图像尺寸统计(体素数)==========")
    print(
        f"median size : D={median_size[0]:.0f}  H={median_size[1]:.0f}  W={median_size[2]:.0f}"
    )
    print(f"min    size : D={min_size[0]}  H={min_size[1]}  W={min_size[2]}")
    print(f"max    size : D={max_size[0]}  H={max_size[1]}  W={max_size[2]}")

    # ── 推荐 target spacing ──────────────────────────────────────
    # nnUNet 的逻辑:
    #   xy 轴(平面内):直接用 median
    #   z  轴(轴向)  :如果 median_z > 2 * median_xy,说明是各向异性数据
    #                    这时候 z 轴不做过度插值,保持原始采集分辨率
    sx, sy, sz = median_spacing[2], median_spacing[1], median_spacing[0]
    in_plane = np.median([sx, sy])

    if sz > 2 * in_plane:
        # 各向异性:z 轴保持 median,xy 轴用 median
        target = (round(sz, 2), round(in_plane, 2), round(in_plane, 2))
        note = "各向异性数据,z 轴不过度插值"
    else:
        # 接近各向同性:三轴都用 median
        target = (round(in_plane, 2), round(in_plane, 2), round(in_plane, 2))
        note = "接近各向同性,三轴统一"

    print("\n========== 推荐 Target Spacing ==========")
    print(f"类型   : {note}")
    print(f"推荐   : {list(target)}")
    print("\n把这个值填进 preprocess_resample.py:")
    print(f"  pixdim = {list(target)}")


if __name__ == "__main__":
    main()
