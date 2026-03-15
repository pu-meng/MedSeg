from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

from typing import Dict, List, Optional, Tuple

import torch
from monai.inferers.utils import sliding_window_inference

from twostage_medseg.twostage.eval_helpers import dice_binary, safe_case_name,add_project_to_syspath
from twostage_medseg.twostage.metrics import build_final_pred_from_liver_tumor, summarize_metric
from  twostage_medseg.data.prepare import prepare_workdir_and_config


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)





def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project_root",
        type=str,
        required=True,
        help="项目根目录, 里面应包含 medseg/ 和 twostage_medseg/",
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
        "--save_pred_pt",
        action="store_true",
        help="保存每个case的预测结果 .pt 文件",
    )
    p.add_argument("--save_dir", type=str, default="./experiments_twostage_eval")
    return p.parse_args()





def build_and_load_models(args, device: str):
    """
    构建并加载两个阶段的模型
    """
    from medseg.models.build_model import build_model
    from medseg.utils.ckpt import load_ckpt

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
    return stage1, stage2


def run_stage1_inference(
    image: torch.Tensor,
    stage1,
    args,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入:
        image: [1,D,H,W]
    返回:
        pred1: [D,H,W] long, 0/1
        liver_mask: [D,H,W] bool
    """
    x = image.unsqueeze(0).to(device)  # [1,1,D,H,W]

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(device == "cuda"),
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
    return pred1, liver_mask


def run_stage2_inference(
    image: torch.Tensor,
    pred1: torch.Tensor,
    liver_mask: torch.Tensor,
    stage2,
    args,
    device: str,
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    输入:
        image: [1,D,H,W]
        pred1: [D,H,W]
        liver_mask: [D,H,W] bool

    返回:
        tumor_full: [D,H,W] long, 0/1
        bbox_info: dict or None
    """
    from twostage_medseg.twostage.roi_utils import (
        compute_bbox_from_mask,
        crop_3d,
        paste_3d,
        bbox_to_dict,
    )

    if liver_mask.sum().item() == 0:
        tumor_full = torch.zeros_like(pred1, dtype=torch.long)
        bbox_info = None
        return tumor_full, bbox_info

    bbox = compute_bbox_from_mask(liver_mask, margin=args.margin)
    image_roi = crop_3d(image, bbox)  # [1,d,h,w]
    x_roi = image_roi.unsqueeze(0).to(device)  # [1,1,d,h,w]

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(device == "cuda"),
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

    return tumor_full, bbox_info


def evaluate_one_case(
    pt_path: str,
    stage1,
    stage2,
    args,
    device: str,
    pred_dir: Optional[str] = None,
) -> Dict:
    """
    对单个 case 做 two-stage 推理与评估
    返回一个 row dict
    """
    case_name = safe_case_name(pt_path)
    data = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)

    image = data["image"].float()  # [1,D,H,W]
    label = data.get("label", None)  # [1,D,H,W] or None

    pred1, liver_mask = run_stage1_inference(
        image=image,
        stage1=stage1,
        args=args,
        device=device,
    )

    tumor_full, bbox_info = run_stage2_inference(
        image=image,
        pred1=pred1,
        liver_mask=liver_mask,
        stage2=stage2,
        args=args,
        device=device,
    )

    tumor_mask = tumor_full == 1
    final_pred = build_final_pred_from_liver_tumor(
        liver_mask=liver_mask,
        tumor_mask=tumor_mask,
    )  # [D,H,W], 0/1/2

    row: Dict = {
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

    if pred_dir is not None:
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

    return row


def print_case_log(i: int, total: int, row: Dict) -> None:
    msg = f"[{i}/{total}] {row['case_name']}"
    if "liver_dice" in row:
        msg += f" liver={row['liver_dice']:.4f} tumor={row['tumor_dice']:.4f}"
    msg += f" tumor_vox={row['pred_tumor_voxels']}"
    print(msg)


def save_per_case_csv(workdir: str, rows: List[Dict]) -> str:
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

    return csv_path


def save_summary_json(workdir: str, rows: List[Dict]) -> str:
    path = os.path.join(workdir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return path


def build_metrics(
    rows: List[Dict],
    liver_dices: List[float],
    tumor_dices: List[float],
    device: str,
    elapsed_hours: float,
) -> Dict:
    return {
        "n_cases": len(rows),
        "device": device,
        "elapsed_hours": round(elapsed_hours, 3),
        "liver_dice": summarize_metric(liver_dices),
        "tumor_dice": summarize_metric(tumor_dices),
    }


def save_metrics_json(workdir: str, metrics: Dict) -> str:
    path = os.path.join(workdir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


def save_report_txt(workdir: str, metrics: Dict) -> str:
    path = os.path.join(workdir, "report.txt")
    with open(path, "w", encoding="utf-8") as f:
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

    return path


def main():
    args = parse_args()
    add_project_to_syspath(args.project_root)

    from medseg.data.dataset_offline import load_pt_paths

    workdir, pred_dir = prepare_workdir_and_config(args)

    pt_paths = load_pt_paths(args.preprocessed_root)
    if args.n > 0:
        pt_paths = pt_paths[: args.n]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1, stage2 = build_and_load_models(args, device)

    rows: List[Dict] = []
    liver_dices: List[float] = []
    tumor_dices: List[float] = []

    time_start = time.time()

    with torch.no_grad():
        for i, pt_path in enumerate(pt_paths, start=1):
            row = evaluate_one_case(
                pt_path=pt_path,
                stage1=stage1,
                stage2=stage2,
                args=args,
                device=device,
                pred_dir=pred_dir,
            )
            rows.append(row)

            if "liver_dice" in row:
                liver_dices.append(float(row["liver_dice"]))
                tumor_dices.append(float(row["tumor_dice"]))

            print_case_log(i, len(pt_paths), row)

    elapsed_hours = (time.time() - time_start) / 3600.0

    save_per_case_csv(workdir, rows)
    save_summary_json(workdir, rows)

    metrics = build_metrics(
        rows=rows,
        liver_dices=liver_dices,
        tumor_dices=tumor_dices,
        device=device,
        elapsed_hours=elapsed_hours,
    )
    save_metrics_json(workdir, metrics)
    save_report_txt(workdir, metrics)

    print("\n===== Final Metrics =====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {workdir}")


if __name__ == "__main__":
    main()
