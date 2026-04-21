"""
calc_ratios.py
统计 Task03_Liver_0p88mm 各类别体素占比,推荐 RandCropByLabelClassesd 的 ratios
用法:python calc_ratios.py --data_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver
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
        default="/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver",
    )
    p.add_argument("--num_classes", type=int, default=3)
    args = p.parse_args()

    lab_paths = sorted(glob.glob(os.path.join(args.data_root, "labelsTr", "*.nii.gz")))
    assert len(lab_paths) > 0, f"找不到标注: {args.data_root}"

    print(f"共 {len(lab_paths)} 个 case,开始统计类别体素占比...")

    # 每个类别的总体素数
    class_counts = np.zeros(args.num_classes, dtype=np.int64)
    # 每个类别出现在多少个 case 里
    class_cases = np.zeros(args.num_classes, dtype=np.int64)

    for lab_p in tqdm(lab_paths):
        lab = nib.load(lab_p).get_fdata(dtype=np.float32)
        for c in range(args.num_classes):
            cnt = int((lab == c).sum())
            class_counts[c] += cnt
            if cnt > 0:
                class_cases[c] += 1

    total_voxels = class_counts.sum()

    print("\n========== 类别统计 ==========")
    class_names = {0: "背景", 1: "肝脏", 2: "肿瘤"}
    for c in range(args.num_classes):
        name = class_names.get(c, f"class{c}")
        ratio = class_counts[c] / total_voxels * 100
        presence = class_cases[c] / len(lab_paths) * 100
        print(
            f"  class {c} ({name}): "
            f"体素占比={ratio:.2f}%  "
            f"出现在 {class_cases[c]}/{len(lab_paths)} 个case ({presence:.1f}%)"
        )

    # ── 计算推荐 ratios ──────────────────────────────────────────
    # nnUNet 逻辑:
    #   背景(0):固定给 0,不以背景为中心采样,没意义
    #   前景类别:体素占比越少 → 给越高的采样权重(成反比)
    #   归一化到加和为 1

    fg_counts = class_counts[1:]  # 只看前景类别(去掉背景)
    fg_counts = np.maximum(fg_counts, 1)  # 防止除零

    # 反比:占比越少,权重越大
    inv = 1.0 / fg_counts.astype(np.float64)
    inv_normalized = inv / inv.sum()  # 归一化

    # 四舍五入到两位小数,保证加和为 1
    ratios_fg = np.round(inv_normalized, 2)
    ratios_fg[-1] = round(1.0 - ratios_fg[:-1].sum(), 2)  # 最后一个补齐误差

    ratios_full = [0.0] + ratios_fg.tolist()  # 背景固定 0

    print("\n========== 推荐 ratios ==========")
    for c, r in enumerate(ratios_full):
        name = class_names.get(c, f"class{c}")
        print(f"  class {c} ({name}): {r}")

    print("\n在 transforms.py / train.py 里使用:")
    print(f"  ratios = {ratios_full}")
    print("\nRandCropByLabelClassesd 参数:")
    print(f"  ratios=list({ratios_full})")

    # ── 额外建议 ────────────────────────────────────────────────
    tumor_presence = class_cases[2] / len(lab_paths)
    if tumor_presence < 0.5:
        print(
            f"\n[注意] 只有 {class_cases[2]} 个 case 有肿瘤({tumor_presence * 100:.1f}%),"
            f"建议同时用 --train_n 筛掉纯阴性 case,或在采样时强制保证 tumor patch 比例."
        )
    else:
        print(
            f"\n[OK] {class_cases[2]} 个 case 有肿瘤({tumor_presence * 100:.1f}%),"
            f"ratios 设置有效."
        )


if __name__ == "__main__":
    main()
