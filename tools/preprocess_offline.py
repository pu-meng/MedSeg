"""
preprocess_offline.py
=====================
离线预处理:对原始 Task03_Liver 数据集做全套确定性预处理:
  1. 方向统一(Orientationd → RAS)
  2. spacing 重采样(Spacingd → 0.88mm 各向同性)
  3. 强度归一化(CT 肝脏窗口 → [0,1])
  4. 前景裁剪(CropForegroundd,去掉大量空气)
  5. 保存为 .pt 文件(torch tensor,训练时直接读取)

使用方式:
  python tools/preprocess_offline.py \
    --data_root /home/pumengyu/Task03_Liver \
    --out_dir   /home/pumengyu/Task03_Liver_pt \
    --num_workers 4

训练时 transforms 只需保留随机增强部分,速度大幅提升.
"""

import os
import glob
import argparse
import torch


from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader

# ── CT 肝脏窗口(与 transforms.py 保持一致)────────────────────────────────
LIVER_WIN_MIN = -13.7
LIVER_WIN_MAX = 188.3

# ── target spacing(与 transforms.py 保持一致)─────────────────────────────
TARGET_SPACING = (0.88, 0.88, 0.88)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        default="/home/pumengyu/Task03_Liver",
        help="原始数据集根目录(含 imagesTr/ 和 labelsTr/)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="/home/pumengyu/Task03_Liver_pt",
        help="输出目录,保存预处理后的 .pt 文件",
    )
    p.add_argument(
        "--num_workers", type=int, default=0, help="DataLoader workers,先用0保证跑通"
    )
    p.add_argument("--overwrite", action="store_true", help="强制重新处理已存在的文件")
    return p.parse_args()


def build_transforms():
    """
    只做确定性操作,不做任何随机增强.
    随机增强留给训练时在线做.

    pipeline 与 transforms.py 的在线流程完全对齐:
      Orientationd → Spacingd → ScaleIntensityRanged → CropForegroundd
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # 统一方向:RAS(与在线 transforms.py 一致)
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # spacing 重采样:原始 CT 体素 → 0.88mm 各向同性
            Spacingd(
                keys=["image", "label"],
                pixdim=TARGET_SPACING,
                mode=("bilinear", "nearest"),
            ),
            # 强度归一化:CT 肝脏窗口 → [0, 1]
            ScaleIntensityRanged(
                keys=["image"],
                a_min=LIVER_WIN_MIN,
                a_max=LIVER_WIN_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # 前景裁剪:去掉大量空气背景,减小体积
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # 转换为训练所需 dtype
            EnsureTyped(
                keys=["image", "label"],
                dtype=(torch.float32, torch.int64),
            ),
        ]
    )


def main():
    args = parse_args()

    # ── 输出目录 ──────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"data_root : {args.data_root}")
    print(f"out_dir   : {args.out_dir}")
    print(f"spacing   : {TARGET_SPACING}")

    # ── 收集待处理样本 ────────────────────────────────────────────────────
    img_paths = sorted(glob.glob(os.path.join(args.data_root, "imagesTr", "*.nii.gz")))
    if len(img_paths) == 0:
        raise FileNotFoundError(f"在 {args.data_root}/imagesTr 下没找到 .nii.gz 文件")

    items = []
    skipped = 0
    for ip in img_paths:
        name = os.path.splitext(os.path.splitext(os.path.basename(ip))[0])[
            0
        ]  # 去掉 .nii.gz
        lp = os.path.join(args.data_root, "labelsTr", os.path.basename(ip))
        if not os.path.exists(lp):
            print(f"[WARN] label 不存在,跳过: {lp}")
            continue

        out_pt = os.path.join(args.out_dir, f"{name}.pt")
        if os.path.exists(out_pt) and not args.overwrite:
            skipped += 1
            continue

        items.append({"image": ip, "label": lp, "name": name})

    print(f"已跳过(已处理): {skipped} 个")
    print(f"待处理: {len(items)} 个")
    if len(items) == 0:
        print("全部已处理完毕,无需重新运行.加 --overwrite 可强制重处理.")
        return

    # ── 构建 Dataset / DataLoader ─────────────────────────────────────────
    tfm = build_transforms()
    ds = Dataset(items, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # ── 逐个处理并保存 ────────────────────────────────────────────────────
    for i, batch in enumerate(loader):
        name = batch["name"][0]

        # shape: [1, C, D, H, W] → squeeze batch dim → [C, D, H, W]
        img = batch["image"][0]  # torch.float32, shape [1, D, H, W]
        lab = batch["label"][0]  # torch.int64,   shape [1, D, H, W]

        out_pt = os.path.join(args.out_dir, f"{name}.pt")
        torch.save({"image": img, "label": lab}, out_pt)

        d, h, w = img.shape[1], img.shape[2], img.shape[3]
        print(f"[{i+1}/{len(items)}] {name}  shape=({d},{h},{w})  -> {out_pt}")

    print("\n✅ 全部处理完成!")
    print(f"共保存 {len(items)} 个 .pt 文件到: {args.out_dir}")



if __name__ == "__main__":
    main()