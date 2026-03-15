from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from monai.inferers.utils import sliding_window_inference


def add_project_to_syspath(project_root: str) -> None:
    project_root = os.path.abspath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project_root",
        type=str,
        required=True,
        help="项目根目录, 里面应包含 medseg/ 和 twostage/",
    )
    p.add_argument("--preprocessed_root", type=str, required=True)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)

    p.add_argument("--stage1_model", type=str, default="dynunet")
    p.add_argument("--stage2_model", type=str, default="dynunet")

    p.add_argument("--stage1_patch", type=int, nargs=3, default=[144, 144, 144])
    p.add_argument("--stage2_patch", type=int, nargs=3, default=[96, 96, 96])

    p.add_argument("--stage1_sw_batch_size", type=int, default=1)
    p.add_argument("--stage2_sw_batch_size", type=int, default=1)

    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--margin", type=int, default=12)

    p.add_argument("--n", type=int, default=0, help="只跑前N个case, 0=全部")
    p.add_argument(
        "--save_pred_pt", action="store_true", help="保存每个case的pt预测结果"
    )
    p.add_argument("--save_dir", type=str, default="./experiments_twostage_eval")
    return p.parse_args()


def dice_binary(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.bool()
    gt = gt.bool()
    inter = (pred & gt).sum().double()
    denom = pred.sum().double() + gt.sum().double()
    return float((2.0 * inter + eps) / (denom + eps))


def safe_case_name(pt_path: str) -> str:
    name = Path(pt_path).name
    if name.endswith(".pt"):
        name = name[:-3]
    return name


def build_final_pred_from_liver_tumor(
    liver_mask: torch.Tensor,
    tumor_mask: torch.Tensor,
) -> torch.Tensor:
    """
    输入:
        liver_mask: [D,H,W] bool
        tumor_mask: [D,H,W] bool
    输出:
        final_pred: [D,H,W] long
            0 background
            1 liver
            2 tumor
    """
    final_pred = torch.zeros_like(liver_mask, dtype=torch.long)
    final_pred[liver_mask] = 1
    final_pred[tumor_mask] = 2
    return final_pred


def summarize_metric(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    x = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": round(float(x.mean().item()), 4),
        "std": round(float(x.std(unbiased=False).item()), 4),
        "min": round(float(x.min().item()), 4),
        "max": round(float(x.max().item()), 4),
    }


def main():
    args = parse_args()
    add_project_to_syspath(args.project_root)

    from medseg.data.dataset_offline import load_pt_paths
    from medseg.models.build_model import build_model
    from medseg.utils.ckpt import load_ckpt
    from twostage_medseg.twostage.roi_utils import (
        compute_bbox_from_mask,
        crop_3d,
        paste_3d,
        bbox_to_dict,
    )

    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    workdir = os.path.join(args.save_dir, timestamp)
    os.makedirs(workdir, exist_ok=True)
    pred_dir = os.path.join(workdir, "predictions")
    if args.save_pred_pt:
        os.makedirs(pred_dir, exist_ok=True)

    config = {
        "project_root": os.path.abspath(args.project_root),
        "preprocessed_root": os.path.abspath(args.preprocessed_root),
        "stage1_ckpt": os.path.abspath(args.stage1_ckpt),
        "stage2_ckpt": os.path.abspath(args.stage2_ckpt),
        "stage1_model": args.stage1_model,
        "stage2_model": args.stage2_model,
        "stage1_patch": list(args.stage1_patch),
        "stage2_patch": list(args.stage2_patch),
        "stage1_sw_batch_size": int(args.stage1_sw_batch_size),
        "stage2_sw_batch_size": int(args.stage2_sw_batch_size),
        "overlap": float(args.overlap),
        "margin": int(args.margin),
        "n": int(args.n),
        "save_pred_pt": bool(args.save_pred_pt),
        "workdir": workdir,
    }
    with open(os.path.join(workdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    pt_paths = load_pt_paths(args.preprocessed_root)
    if args.n > 0:
        pt_paths = pt_paths[: args.n]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    stage1 = build_model(
        args.stage1_model,
        in_channels=1,
        out_channels=2,
        img_size=tuple(args.stage1_patch),
    ).to(device)

    stage2 = build_model(
        args.stage2_model,
        in_channels=1,
        out_channels=2,
        img_size=tuple(args.stage2_patch),
    ).to(device)

    load_ckpt(args.stage1_ckpt, stage1, optimizer=None, map_location=device)
    load_ckpt(args.stage2_ckpt, stage2, optimizer=None, map_location=device)
    stage1.eval()
    stage2.eval()

    rows: List[Dict] = []
    liver_dices: List[float] = []
    tumor_dices: List[float] = []

    time_start = time.time()

    with torch.no_grad():
        for i, pt_path in enumerate(pt_paths, start=1):
            case_name = safe_case_name(pt_path)
            data = torch.load(
                pt_path, map_location="cpu", weights_only=False, mmap=True
            )

            image = data["image"].float()  # [1,D,H,W]
            label = data.get("label", None)  # [1,D,H,W] or None

            x = image.unsqueeze(0).to(device)  # [1,1,D,H,W]

            # -----------------------------
            # stage1: whole-volume liver
            # -----------------------------
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")
            ):
                logits1 = sliding_window_inference(
                    inputs=x,
                    roi_size=tuple(args.stage1_patch),
                    sw_batch_size=args.stage1_sw_batch_size,
                    predictor=stage1,
                    overlap=args.overlap,
                )

            if isinstance(logits1, (tuple, list)):
                logits1 = logits1[0]

            logits1 = torch.as_tensor(logits1)
            pred1 = torch.argmax(logits1.float(), dim=1)[0].cpu()  # [D,H,W]
            liver_mask = pred1 == 1

            # -----------------------------
            # stage2: tumor in liver ROI
            # -----------------------------
            if liver_mask.sum().item() == 0:
                tumor_full = torch.zeros_like(pred1, dtype=torch.long)
                bbox_info: Optional[Dict] = None
            else:
                bbox = compute_bbox_from_mask(liver_mask, margin=args.margin)
                image_roi = crop_3d(image, bbox)  # [1,d,h,w]
                x_roi = image_roi.unsqueeze(0).to(device)  # [1,1,d,h,w]

                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")
                ):
                    logits2 = sliding_window_inference(
                        inputs=x_roi,
                        roi_size=tuple(args.stage2_patch),
                        sw_batch_size=args.stage2_sw_batch_size,
                        predictor=stage2,
                        overlap=args.overlap,
                    )

                if isinstance(logits2, (tuple, list)):
                    logits2 = logits2[0]

                logits2 = torch.as_tensor(logits2)
                pred2 = torch.argmax(logits2.float(), dim=1)[0].cpu()  # [d,h,w], 0/1

                tumor_full = torch.zeros_like(pred1, dtype=torch.long)
                tumor_full = paste_3d(tumor_full, pred2.long(), bbox)
                bbox_info = bbox_to_dict(bbox)

            tumor_mask = tumor_full == 1

            # -----------------------------
            # final whole-volume prediction
            # -----------------------------
            final_pred = build_final_pred_from_liver_tumor(
                liver_mask=liver_mask,
                tumor_mask=tumor_mask,
            )  # [D,H,W], 0/1/2

            row = {
                "case_name": case_name,
                "source_pt": pt_path,
                "pred_liver_voxels": int(liver_mask.sum().item()),
                "pred_tumor_voxels": int(tumor_mask.sum().item()),
                "bbox": bbox_info,
            }

            if label is not None:
                gt = label[0].long()  # [D,H,W], 0/1/2
                gt_liver = gt > 0
                gt_tumor = gt == 2

                liver_dice = dice_binary(final_pred > 0, gt_liver)
                tumor_dice = dice_binary(final_pred == 2, gt_tumor)

                row["liver_dice"] = round(liver_dice, 4)
                row["tumor_dice"] = round(tumor_dice, 4)

                liver_dices.append(liver_dice)
                tumor_dices.append(tumor_dice)

            if args.save_pred_pt:
                torch.save(
                    {
                        "image": image,  # [1,D,H,W]
                        "label": label,  # [1,D,H,W] or None
                        "stage1_liver_pred": pred1.unsqueeze(0).long(),
                        "stage2_tumor_pred": tumor_full.unsqueeze(0).long(),
                        "final_pred": final_pred.unsqueeze(0).long(),
                        "meta": row,
                    },
                    os.path.join(pred_dir, f"{case_name}_twostage.pt"),
                )

            rows.append(row)

            msg = f"[{i}/{len(pt_paths)}] {case_name}"
            if "liver_dice" in row:
                msg += f" liver={row['liver_dice']:.4f} tumor={row['tumor_dice']:.4f}"
            msg += f" tumor_vox={row['pred_tumor_voxels']}"
            print(msg)

    # -----------------------------
    # save per-case csv
    # -----------------------------
    csv_path = os.path.join(workdir, "per_case.csv")
    fieldnames = [
        "case_name",
        "source_pt",
        "pred_liver_voxels",
        "pred_tumor_voxels",
        "liver_dice",
        "tumor_dice",
        "bbox",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            rr = dict(r)
            rr["bbox"] = (
                json.dumps(rr["bbox"], ensure_ascii=False)
                if rr["bbox"] is not None
                else ""
            )
            writer.writerow({k: rr.get(k, "") for k in fieldnames})

    # -----------------------------
    # save summary / metrics
    # -----------------------------
    with open(os.path.join(workdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    metrics = {
        "n_cases": len(rows),
        "device": device,
        "elapsed_hours": round((time.time() - time_start) / 3600.0, 3),
        "liver_dice": summarize_metric(liver_dices),
        "tumor_dice": summarize_metric(tumor_dices),
    }

    with open(os.path.join(workdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(workdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("Two-Stage Evaluation Report\n")
        f.write("===========================\n")
        f.write(f"workdir: {workdir}\n")
        f.write(f"n_cases: {metrics['n_cases']}\n")
        f.write(f"device: {metrics['device']}\n")
        f.write(f"elapsed_hours: {metrics['elapsed_hours']}\n\n")

        f.write("Liver Dice:\n")
        for k, v in metrics["liver_dice"].items():
            f.write(f"  {k}: {v}\n")

        f.write("\nTumor Dice:\n")
        for k, v in metrics["tumor_dice"].items():
            f.write(f"  {k}: {v}\n")

    print("\n===== Final Metrics =====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {workdir}")


if __name__ == "__main__":
    main()
