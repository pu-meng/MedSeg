"""
calc_ratios_nnunet.py
=====================
用 nnUNet 思想统计数据集分布,自动推荐两阶段 ratios.

支持两种输入:
  1. 原始 .nii.gz 标注文件
  2. 已预处理的 .pt 文件(更快)

用法:
  # 用 .pt 文件统计(推荐,快)
  python calc_ratios_nnunet.py --pt_dir /path/to/preprocessed_pt

  # 用原始 .nii.gz 统计
  python calc_ratios_nnunet.py --data_root /path/to/Task03_Liver_0.88mm

输出:
  - 每个类别的体素占比,case出现率
  - 推荐的 stage1 ratios(前1/3 epoch,稳liver)
  - 推荐的 stage2 ratios(后2/3 epoch,补tumor)
  - 直接可以粘贴进 train.py 的代码
"""

import os
import glob
import argparse
import numpy as np

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(x, **kwargs):
        return x


def load_label_from_pt(pt_path):
    """从 .pt 文件加载 label tensor,转为 numpy"""
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    lab = data["label"]
    # shape: [1, D, H, W] 或 [D, H, W]
    if lab.ndim == 4:
        lab = lab[0]
    return lab.numpy().astype(np.int32)


def load_label_from_nii(nii_path):
    """从 .nii.gz 文件加载 label,转为 numpy"""
    import nibabel as nib

    return nib.load(nii_path).get_fdata(dtype=np.float32).astype(np.int32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pt_dir", type=str, default=None, help="预处理 .pt 文件目录(优先使用)"
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="原始数据集目录,含 labelsTr/*.nii.gz",
    )
    p.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="类别数(包含背景),默认3(bg/liver/tumor)",
    )
    p.add_argument(
        "--class_names", type=str, default="背景,肝脏,肿瘤", help="类别名称,逗号分隔"
    )
    return p.parse_args()


def compute_stats(label_paths, load_fn, num_classes):
    """
    统计每个类别的:
      - 总体素数
      - 出现的case数
      - 每个case的平均体素数(仅在出现时)
    """
    class_total_vox = np.zeros(num_classes, dtype=np.int64)
    class_case_count = np.zeros(num_classes, dtype=np.int64)
    class_vox_when_present = [[] for _ in range(num_classes)]

    for path in tqdm(label_paths, desc="统计中"):
        lab = load_fn(path)

        for c in range(num_classes):
            cnt = int((lab == c).sum())
            class_total_vox[c] += cnt
            if cnt > 0:
                class_case_count[c] += 1
                class_vox_when_present[c].append(cnt)

    return class_total_vox, class_case_count, class_vox_when_present


def recommend_ratios(class_total_vox, class_case_count, n_cases, num_classes):
    """
    nnUNet 核心思想:
      1. 背景(class 0) 固定为 0(不以背景为中心采样)
      2. 前景类别:综合考虑[体素占比]和[case出现率]
         - 体素占比越低 → 权重越高(需要更多采样才能见到)
         - case出现率越低 → 适当惩罚(没有tumor的case采也没用)
      3. 两阶段:
         - Stage1(前1/3 epoch):以主器官(liver)为主,tumor占少量
           目标:让模型先学会分割大目标,建立空间感知
         - Stage2(后2/3 epoch):以小目标(tumor)为主
           目标:在liver已稳的基础上,专攻tumor recall

    Returns:
        stage1_ratios, stage2_ratios: 包含背景0的完整ratios列表
    """
    fg_vox = class_total_vox[1:].astype(np.float64)
    fg_cases = class_case_count[1:].astype(np.float64)
    n_fg = len(fg_vox)

    # 保护:防止除零
    fg_vox = np.maximum(fg_vox, 1.0)
    fg_cases = np.maximum(fg_cases, 1.0)

    # 体素占比
    vox_ratio = fg_vox / fg_vox.sum()

    # case出现率(作为权重上限的约束)
    case_presence = fg_cases / n_cases  # [0,1]

    # ── Stage1:主器官优先(liver先稳)──────────────────────────────
    # 思路:给liver更高权重,tumor给一点点但不过多
    # 方法:直接按体素占比反比,但给最大类别(liver)额外加权
    inv_vox = 1.0 / vox_ratio
    inv_normalized = inv_vox / inv_vox.sum()

    # Stage1:把最大类别权重抬高(让模型先学好它)
    # 找最大体素类别(通常是liver=index 0 in fg)
    biggest_fg_idx = int(np.argmax(fg_vox))  # 通常是0(liver)
    s1 = inv_normalized.copy()
    # 重新分配:biggest类给50%,其余按反比分剩余50%
    s1_boost = np.zeros(n_fg)
    s1_boost[biggest_fg_idx] = 0.5
    others_idx = [i for i in range(n_fg) if i != biggest_fg_idx]
    if others_idx:
        others_inv = inv_normalized[others_idx]
        others_inv = others_inv / others_inv.sum() * 0.5
        for i, idx in enumerate(others_idx):
            s1_boost[idx] = others_inv[i]
    s1 = s1_boost

    # ── Stage2:小目标优先(tumor加强)──────────────────────────────
    # 思路:给最稀有类别(tumor)更高权重
    # 但受case出现率约束:如果只有90%的case有tumor,tumor权重上限~0.9
    smallest_fg_idx = int(np.argmin(fg_vox))  # 通常是1(tumor)
    s2 = np.zeros(n_fg)
    # 给最小类别 60%,其余按体素大小正比分剩余 40%
    s2[smallest_fg_idx] = min(0.6, float(case_presence[smallest_fg_idx]))
    remaining = 1.0 - s2[smallest_fg_idx]
    others_idx2 = [i for i in range(n_fg) if i != smallest_fg_idx]
    if others_idx2:
        others_vox = fg_vox[others_idx2]
        others_prop = others_vox / others_vox.sum() * remaining
        for i, idx in enumerate(others_idx2):
            s2[idx] = others_prop[i]

    # 归一化,保证加和为1
    s1 = s1 / s1.sum()
    s2 = s2 / s2.sum()

    # 保留2位小数
    s1 = np.round(s1, 2)
    s2 = np.round(s2, 2)
    # 补齐浮点误差(确保加和精确等于1)
    s1[-1] = round(1.0 - float(s1[:-1].sum()), 2)
    s2[-1] = round(1.0 - float(s2[:-1].sum()), 2)

    # 拼上背景0
    stage1_ratios = [0.0] + s1.tolist()
    stage2_ratios = [0.0] + s2.tolist()

    return stage1_ratios, stage2_ratios


def print_report(
    class_total_vox,
    class_case_count,
    class_vox_when_present,
    n_cases,
    num_classes,
    class_names,
    stage1,
    stage2,
):

    total_vox = class_total_vox.sum()

    print("\n" + "=" * 55)
    print("         数据集类别统计(nnUNet风格)")
    print("=" * 55)
    print(
        f"{'类别':<6} {'名称':<8} {'体素占比':>8} {'出现case':>10} {'平均体素/case':>14}"
    )
    print("-" * 55)
    for c in range(num_classes):
        name = class_names[c] if c < len(class_names) else f"class{c}"
        ratio = class_total_vox[c] / total_vox * 100
        presence = class_case_count[c]
        vox_list = class_vox_when_present[c]
        avg_vox = int(np.mean(vox_list)) if vox_list else 0
        print(
            f"  {c:<4} {name:<8} {ratio:>7.2f}%  {presence:>4}/{n_cases}case  {avg_vox:>12,}"
        )

    print("=" * 55)

    print("\n" + "=" * 55)
    print("         推荐 ratios(nnUNet两阶段思想)")
    print("=" * 55)
    print("\n[Stage1 ratios]前1/3 epoch,先稳住主器官(liver):")
    for c, r in enumerate(stage1):
        name = class_names[c] if c < len(class_names) else f"class{c}"
        print(f"    class {c} ({name}): {r}")

    print("\n[Stage2 ratios]后2/3 epoch,专攻小目标(tumor):")
    for c, r in enumerate(stage2):
        name = class_names[c] if c < len(class_names) else f"class{c}"
        print(f"    class {c} ({name}): {r}")

    print("\n" + "=" * 55)
    print("  直接粘贴进 train.py 的 get_stage_ratios 函数:")
    print("=" * 55)
    print(f"""
def get_stage_ratios(epoch: int, epochs: int):
    cut = int(epochs / 3)
    if epoch <= cut:
        # 前1/3:先学好主器官(liver)
        return {tuple(stage1)}
    else:
        # 后2/3:专攻小目标(tumor)
        return {tuple(stage2)}
""")

    # 额外诊断建议
    tumor_presence_rate = class_case_count[2] / n_cases if num_classes > 2 else 0
    print("=" * 55)
    print("  额外建议")
    print("=" * 55)
    if num_classes > 2:
        tumor_avg = (
            int(np.mean(class_vox_when_present[2])) if class_vox_when_present[2] else 0
        )
        liver_avg = (
            int(np.mean(class_vox_when_present[1])) if class_vox_when_present[1] else 0
        )
        ratio_t2l = tumor_avg / max(liver_avg, 1)
        print(
            f"\n  Tumor/Liver 平均体素比 = {ratio_t2l:.4f}  ({tumor_avg:,} / {liver_avg:,})"
        )
        if ratio_t2l < 0.05:
            print("  ⚠️  tumor极小(<5%的liver大小),建议同时:")
            print("      1. loss 用 FocalTversky(alpha=0.3, beta=0.7)")
            print("      2. 推理后处理:tumor只在dilate(pred_liver)内允许出现")
            print("      3. 评估时分别报告[全集]和[有tumor子集]的dice")
        if tumor_presence_rate < 0.95:
            print(f"\n  ⚠️  {int((1 - tumor_presence_rate) * n_cases)}个case没有tumor,")
            print(
                "      Stage2的tumor权重建议不超过 {:.2f}".format(tumor_presence_rate)
            )


def main():
    args = parse_args()
    class_names = args.class_names.split(",")

    # 确定数据来源
    if args.pt_dir:
        label_paths = sorted(glob.glob(os.path.join(args.pt_dir, "*.pt")))
        if not label_paths:
            raise FileNotFoundError(f"找不到 .pt 文件: {args.pt_dir}")
        load_fn = load_label_from_pt
        print(f"[模式] 读取 .pt 文件,共 {len(label_paths)} 个")
    elif args.data_root:
        label_paths = sorted(
            glob.glob(os.path.join(args.data_root, "labelsTr", "*.nii.gz"))
        )
        if not label_paths:
            raise FileNotFoundError(f"找不到标注: {args.data_root}/labelsTr")
        load_fn = load_label_from_nii
        print(f"[模式] 读取 .nii.gz 文件,共 {len(label_paths)} 个")
    else:
        raise ValueError("请传入 --pt_dir 或 --data_root")

    n_cases = len(label_paths)
    class_total_vox, class_case_count, class_vox_when_present = compute_stats(
        label_paths, load_fn, args.num_classes
    )

    stage1, stage2 = recommend_ratios(
        class_total_vox, class_case_count, n_cases, args.num_classes
    )

    print_report(
        class_total_vox,
        class_case_count,
        class_vox_when_present,
        n_cases,
        args.num_classes,
        class_names,
        stage1,
        stage2,
    )


if __name__ == "__main__":
    main()
