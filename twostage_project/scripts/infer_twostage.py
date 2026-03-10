"""
scripts/infer_twostage.py

两阶段联合推理:
  Stage1: 全CT → liver mask
  Stage2: liver ROI → tumor mask
  合并:   0=背景, 1=肝脏, 2=肿瘤

用法:
    python scripts/infer_twostage.py \
        --data_root /home/pumengyu/Task03_Liver \
        --stage1_ckpt /home/pumengyu/experiments/liver_v1/train/xx/best.pt \
        --stage2_ckpt /home/pumengyu/experiments/tumor_v1/train/xx/best.pt \
        --output_dir /home/pumengyu/predictions
"""

import os
import sys
import argparse
import numpy as np
import torch

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medseg.models.build_model import build_model
from medseg.data.msd import load_msd_dataset

WIN_MIN = -13.7
WIN_MAX = 188.3


# ─────────────────────────────────────────────
# 预处理
# ─────────────────────────────────────────────


def get_preprocess():
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(0.88, 0.88, 0.88), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=WIN_MIN,
                a_max=WIN_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )


# ─────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────


def load_model(ckpt_path, out_channels, device):
    model = build_model(
        "unet3d",
        in_channels=1,
        out_channels=out_channels,
        img_size=(96, 96, 96),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  加载完成: {ckpt_path}")
    return model


# ─────────────────────────────────────────────
# 单样本推理
# ─────────────────────────────────────────────


@torch.no_grad()
def predict(model, x, roi_size, sw_batch_size, overlap, device):
    """
    roi_size: sliding window 的输入尺寸, 需要根据模型训练时的 patch_size 来设置
    也就是sliding_window_inference中每次处理的子块大小, 过大可能导致显存不足, 过小可能导致预测不准确
    overlap: sliding window 的重叠比例, 取值范围 [0, 1), 过大可能导致推理时间增加, 过小可能导致边界
    步长=roi_size*(1-overlap), 过大可能导致预测不连续, 过小可能导致推理时间增加

    窗口数量是由大小和overlap自动决定的，不需要手动计算
    sw_batch_size 是每次送入模型的窗口数量, 过大可能导致显存不足, 过小可能导致推理时间增加
    """
    x = x.to(device)
    logits = sliding_window_inference(
        inputs=x,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()


def infer_one(
    image_path,
    stage1_model,
    stage2_model,
    device,
    roi_size_s1,
    roi_size_s2,
    sw_batch_size,
    overlap,
    margin=20,
):

    tf = get_preprocess()
    data = tf({"image": image_path})
    x = data["image"].unsqueeze(0)  # (1, 1, Z, Y, X)

    # ── Stage1: 全图预测 liver ──
    s1_pred = predict(stage1_model, x, roi_size_s1, sw_batch_size, overlap, device)
    # s1_pred shape: (Z, Y, X), 值 0/1/2

    liver_mask = s1_pred >= 1
    if liver_mask.sum() == 0:
        print("  [warn] Stage1 未检测到肝脏, 直接返回 Stage1 结果")
        return s1_pred

    # ── 裁剪 liver ROI ──
    coords = np.where(liver_mask)
    mins = [max(0, c.min() - margin) for c in coords]
    maxs = [min(s - 1, c.max() + margin) for c, s in zip(coords, s1_pred.shape)]

    z0, y0, x0 = mins
    z1, y1, x1 = [m + 1 for m in maxs]

    x_roi = x[:, :, z0:z1, y0:y1, x0:x1]  # (1, 1, rZ, rY, rX)
    print(f"  ROI shape={x_roi.shape[2:]}")

    # ── Stage2: ROI 内预测 tumor ──
    s2_pred = predict(stage2_model, x_roi, roi_size_s2, sw_batch_size, overlap, device)
    # s2_pred shape: (rZ, rY, rX), 值 0/1 (1=肿瘤)

    # ── 合并结果 ──
    final = s1_pred.copy()  # 保留 Stage1 的 liver (值=1)
    final[final == 2] = 0  # 清掉 Stage1 的肿瘤预测(用 Stage2 替换)

    tumor_roi = np.zeros_like(s1_pred)
    tumor_roi[z0:z1, y0:y1, x0:x1] = s2_pred
    final[tumor_roi == 1] = 2  # 写入 Stage2 的肿瘤预测

    return final


def compute_dice(pred, gt_path):
    """读取 GT 并重采样到和 pred 一致的空间后计算 Dice"""
    from monai.transforms import (
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        EnsureTyped,
    )
    import torch

    # 对 GT label 做和推理时一样的预处理
    tf = Compose(
        [
            LoadImaged(keys=["label"]),
            EnsureChannelFirstd(keys=["label"]),
            Orientationd(keys=["label"], axcodes="RAS"),
            Spacingd(keys=["label"], pixdim=(0.88, 0.88, 0.88), mode="nearest"),
            EnsureTyped(keys=["label"], dtype=torch.int64),
        ]
    )
    gt_arr = tf({"label": gt_path})["label"][0].numpy()

    eps = 1e-8
    results = {}
    for c, name in [(1, "liver"), (2, "tumor")]:
        p = pred == c
        g = gt_arr == c
        inter = (p & g).sum()
        denom = p.sum() + g.sum()
        dice = (2.0 * inter + eps) / (denom + eps)
        results[name] = round(float(dice), 4)

    return results


# ─────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)

    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--roi_s1", type=int, nargs=3, default=[144, 144, 144])
    p.add_argument("--roi_s2", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--sw_batch_size", type=int, default=2)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--margin", type=int, default=20)
    return p.parse_args()


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("加载 Stage1 模型 (liver, 3类)...")
    s1_model = load_model(args.stage1_ckpt, out_channels=3, device=device)

    print("加载 Stage2 模型 (tumor, 2类)...")
    s2_model = load_model(args.stage2_ckpt, out_channels=2, device=device)

    train_items, test_items = load_msd_dataset(args.data_root)
    items = train_items if args.split == "train" else test_items
    print(f"\n推理 {len(items)} 个样本 ({args.split})\n")

    all_liver, all_tumor = [], []

    for item in items:
        sid = item["id"]
        print(f"── {sid}")

        pred = infer_one(
            item["image"],
            s1_model,
            s2_model,
            device=device,
            roi_size_s1=tuple(args.roi_s1),
            roi_size_s2=tuple(args.roi_s2),
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            margin=args.margin,
        )

        dice = compute_dice(pred, item["label"])
        all_liver.append(dice["liver"])
        all_tumor.append(dice["tumor"])
        print(f"  liver={dice['liver']:.4f}  tumor={dice['tumor']:.4f}")

    # 汇总
    n = len(all_liver)
    print(f"\n{'=' * 40}")
    print(f"样本数:         {n}")
    print(f"平均 liver_dice = {sum(all_liver) / n:.4f}")
    print(f"平均 tumor_dice = {sum(all_tumor) / n:.4f}")


if __name__ == "__main__":
    main()
