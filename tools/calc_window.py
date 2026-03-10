"""
calc_window.py
统计 Task03_Liver_1p5mm 数据集的 intensity window 参数
输出:window_min, window_max, mean, std
用法:python calc_window.py --data_root /path/to/Task03_Liver_1p5mm
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        default="/home/pumengyu/Task03_Liver",
    )
    args = p.parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.data_root, "imagesTr", "*.nii.gz")))
    lab_paths = sorted(glob.glob(os.path.join(args.data_root, "labelsTr", "*.nii.gz")))
    assert len(img_paths) == len(lab_paths) and len(img_paths) > 0, (
        f"找不到数据或数量不匹配: {args.data_root}"
    )

    print(f"共 {len(img_paths)} 个 case,开始统计前景强度...")

    all_fg = []  # 收集所有 case 的前景体素强度值

    for img_p, lab_p in tqdm(zip(img_paths, lab_paths), total=len(img_paths)):
        img = nib.load(img_p).get_fdata(dtype=np.float32)  # [D, H, W]
        lab = nib.load(lab_p).get_fdata(dtype=np.float32)

        fg_mask = lab > 0  # 前景 = 肝脏(1) + 肿瘤(2)
        fg_voxels = img[fg_mask]  # 只取前景区域的强度值

        if len(fg_voxels) == 0:
            print(f"  [warn] 跳过无前景 case: {os.path.basename(img_p)}")
            continue

        all_fg.append(fg_voxels)

    all_fg = np.concatenate(all_fg)  # 合并所有 case

    window_min = float(np.percentile(all_fg, 0.5))
    window_max = float(np.percentile(all_fg, 99.5))
    mean = float(np.mean(all_fg))
    std = float(np.std(all_fg))

    print("\n========== 统计结果 ==========")
    print(f"总前景体素数:  {len(all_fg):,}")
    print(f"window_min  =  {window_min:.2f}   (0.5% 分位数)")
    print(f"window_max  =  {window_max:.2f}   (99.5% 分位数)")
    print(f"mean        =  {mean:.2f}")
    print(f"std         =  {std:.2f}")
    print("================================")
    print("\n把上面的值填进 transforms.py:")
    print(f"  LIVER_WIN_MIN = {window_min:.1f}")
    print(f"  LIVER_WIN_MAX = {window_max:.1f}")


if __name__ == "__main__":
    main()
