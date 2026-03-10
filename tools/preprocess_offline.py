"""
preprocess_offline.py
=====================
离线预处理:在已有的 0.88mm 数据集基础上,做:
  1. 强度归一化(CT肝脏窗口)
  2. 前景裁剪(CropForeground)
  3. 保存为 .pt 文件(torch tensor,直接训练读取)

使用方式:
  python preprocess_offline.py \
    --src /home/pumengyu/.../Task03_Liver_0.88mm \
    --dst /home/pumengyu/.../Task03_Liver_0.88mm_preprocessed \
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
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader

# ── CT 肝脏窗口(与 build_transforms.py 保持一致)──────────────────────────
LIVER_WIN_MIN = -13.7
LIVER_WIN_MAX = 188.3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=str,
        default="/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm",
        help="输入数据集根目录(已完成 spacing 预处理的 .nii.gz)",
    )
    p.add_argument(
        "--dst",
        type=str,
        default="/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm_preprocessed",
        help="输出目录,保存 .pt 文件",
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
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # 强度归一化:CT肝脏窗口 → [0, 1]
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
    os.makedirs(args.dst, exist_ok=True)
    print(f"src : {args.src}")
    print(f"dst : {args.dst}")

    # ── 收集待处理样本 ────────────────────────────────────────────────────
    img_paths = sorted(glob.glob(os.path.join(args.src, "imagesTr", "*.nii.gz")))
    if len(img_paths) == 0:
        raise FileNotFoundError(f"在 {args.src}/imagesTr 下没找到 .nii.gz 文件")

    items = []
    skipped = 0
    for ip in img_paths:
        name = os.path.splitext(os.path.splitext(os.path.basename(ip))[0])[
            0
        ]  # 去掉 .nii.gz
        lp = os.path.join(args.src, "labelsTr", os.path.basename(ip))
        if not os.path.exists(lp):
            print(f"[WARN] label 不存在,跳过: {lp}")
            continue

        out_pt = os.path.join(args.dst, f"{name}.pt")
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

        out_pt = os.path.join(args.dst, f"{name}.pt")
        torch.save({"image": img, "label": lab}, out_pt)

        d, h, w = img.shape[1], img.shape[2], img.shape[3]
        print(f"[{i+1}/{len(items)}] {name}  shape=({d},{h},{w})  -> {out_pt}")

    print("\n✅ 全部处理完成!")
    print(f"共保存 {len(items)} 个 .pt 文件到: {args.dst}")
    print()
    print("── 下一步 ──────────────────────────────────────────────────────")
    print("训练时 build_train_transforms 只保留随机增强部分:")
    print("  RandFlipd / RandRotate90d / RandCropByLabelClassesd")
    print("  RandGaussianNoised / RandAdjustContrastd 等")
    print("去掉 LoadImaged / ScaleIntensityRanged / CropForegroundd")
    print("改用自定义 Dataset 直接 torch.load() 读取 .pt 文件即可.")


if __name__ == "__main__":
    main()
