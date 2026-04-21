"""
calc_ct_intensity_stats.py
==========================
计算 Task03_Liver 数据集的 CT foreground 强度统计值，
用于 nnUNet 风格的 CTNormalization。

foreground 定义：label > 0（肝脏+肿瘤区域）

输出：
  percentile_00_5, percentile_99_5, mean, std

用法：
  python tools/calc_ct_intensity_stats.py \
    --data_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver
"""

import argparse
import glob
import os

import numpy as np
import nibabel as nib
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver")
    p.add_argument("--num_samples", type=int, default=10000,
                   help="每个 case 采样的 foreground voxel 数量")
    return p.parse_args()


def main():
    args = parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.data_root, "imagesTr", "*.nii.gz")))
    print(f"共 {len(img_paths)} 个 case")

    all_fg_voxels = []
    rng = np.random.RandomState(1234)

    for img_path in tqdm(img_paths, desc="收集 foreground voxels"):
        name = os.path.basename(img_path)
        lab_path = os.path.join(args.data_root, "labelsTr", name)
        if not os.path.exists(lab_path):
            print(f"[WARN] label 不存在，跳过: {lab_path}")
            continue

        img = nib.load(img_path).get_fdata(dtype=np.float32)
        lab = nib.load(lab_path).get_fdata(dtype=np.float32)

        fg_mask = lab > 0
        fg_voxels = img[fg_mask]

        if len(fg_voxels) == 0:
            continue

        # 采样，避免内存爆炸
        n = min(args.num_samples, len(fg_voxels))
        sampled = rng.choice(fg_voxels, n, replace=False)
        all_fg_voxels.append(sampled)

    all_fg_voxels = np.concatenate(all_fg_voxels)
    print(f"\n总 foreground 采样 voxel 数: {len(all_fg_voxels)}")

    p005 = float(np.percentile(all_fg_voxels, 0.5))
    p995 = float(np.percentile(all_fg_voxels, 99.5))
    mean = float(np.mean(all_fg_voxels))
    std  = float(np.std(all_fg_voxels))

    print(f"\n=== Task03_Liver CT 强度统计（foreground） ===")
    print(f"  percentile_00_5 : {p005:.2f}")
    print(f"  percentile_99_5 : {p995:.2f}")
    print(f"  mean            : {mean:.2f}")
    print(f"  std             : {std:.2f}")
    print(f"\n用于 preprocess_offline.py 的参数：")
    print(f"  CT_CLIP_MIN = {p005:.2f}")
    print(f"  CT_CLIP_MAX = {p995:.2f}")
    print(f"  CT_MEAN     = {mean:.2f}")
    print(f"  CT_STD      = {std:.2f}")


if __name__ == "__main__":
    main()
