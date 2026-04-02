from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from typing import Dict, List

import scipy.ndimage as ndi
import torch
from monai.inferers.utils import sliding_window_inference

from medseg.models.build_model import build_model
from medseg.utils.ckpt import load_ckpt
from medseg.data.dataset_offline import split_three_ways



from twostage_medseg.metrics.filter import filter_largest_component
from twostage_medseg.metrics.metrics_utils import (
    compute_metrics,
    summarize_metrics_list,
)
from twostage_medseg.twostage.vis_utils import save_case_visualization

_twostage_root = "/home/pumengyu"
if _twostage_root not in sys.path:
    sys.path.insert(0, _twostage_root)


# ------------------------------------------------------------------ #
# args
# ------------------------------------------------------------------ #


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--preprocessed_root", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--patch", type=int, nargs=3, default=None)
    p.add_argument("--sw_batch_size", type=int, default=None)
    p.add_argument("--overlap", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--test_ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
    )
    p.add_argument("--n", type=int, default=0, help="只跑前N个case,0=全部")
    p.add_argument("--min_tumor_size", type=int, default=100)
    p.add_argument(
        "--tta", action="store_true", help="测试时增强:8种翻转组合推理取平均"
    )
    p.add_argument("--save_vis", action="store_true", help="保存可视化PNG")
    p.add_argument("--vis_n", type=int, default=10, help="最多保存前N个case的可视化")
    p.add_argument(
        "--save_pred_pt", action="store_true", help="保存每个case的预测结果.pt"
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="结果保存目录，默认自动推导为 <exp_dir>/eval/<timestamp>",
    )
    return p.parse_args()


def resolve_args(args):
    ckpt_abs = os.path.abspath(args.ckpt)
    run_dir = os.path.dirname(ckpt_abs)              # .../train/<timestamp>
    timestamp = os.path.basename(run_dir)
    exp_dir = os.path.dirname(os.path.dirname(run_dir))  # .../exp_name

    if args.preprocessed_root is None:
        raise ValueError("必须传 --preprocessed_root")
    if args.model is None:
        raise ValueError("必须传 --model")
    if args.num_classes is None:
        raise ValueError("必须传 --num_classes")
    if args.patch is None:
        raise ValueError("必须传 --patch")
    if args.overlap is None:
        raise ValueError("必须传 --overlap")
    if args.val_ratio is None:
        raise ValueError("必须传 --val_ratio")
    if args.test_ratio is None:
        raise ValueError("必须传 --test_ratio")
    if args.seed is None:
        raise ValueError("必须传 --seed")

    if args.save_dir is None:
        args.save_dir = os.path.join(exp_dir, "eval", timestamp)
    os.makedirs(args.save_dir, exist_ok=True)

    return args, timestamp


# ------------------------------------------------------------------ #
# inference helpers
# ------------------------------------------------------------------ #


def tta_infer(inputs, roi_size, sw_batch_size, predictor, overlap):
    spatial_axes = [2, 3, 4]
    flip_combos = [[]]
    for ax in spatial_axes:
        flip_combos += [c + [ax] for c in flip_combos]

    logits_sum = None
    for axes in flip_combos:
        x = torch.flip(inputs, dims=axes) if axes else inputs
        out = sliding_window_inference(x, roi_size, sw_batch_size, predictor, overlap)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = torch.as_tensor(out).float()
        if axes:
            out = torch.flip(out, dims=axes)
        logits_sum = out if logits_sum is None else logits_sum + out
    return logits_sum / len(flip_combos)


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #


def main():
    args = parse_args()
    args, timestamp = resolve_args(args)

    workdir = args.save_dir
    pred_dir = None
    vis_dir = None
    if args.save_pred_pt:
        pred_dir = os.path.join(workdir, "pred_pt")
        os.makedirs(pred_dir, exist_ok=True)
    if args.save_vis:
        vis_dir = os.path.join(workdir, "vis_png")
        os.makedirs(vis_dir, exist_ok=True)

    # 保存运行命令
    with open(os.path.join(workdir, "command.txt"), "w", encoding="utf-8") as f:
        cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        prefix = f"CUDA_VISIBLE_DEVICES={cuda_dev} " if cuda_dev else ""
        argv = sys.argv[:]
        first_line = f"{prefix}{sys.executable} {argv[0]}"
        lines = [first_line]
        i = 1
        while i < len(argv):
            tok = argv[i]
            if (
                tok.startswith("--")
                and i + 1 < len(argv)
                and not argv[i + 1].startswith("--")
            ):
                vals = []
                j = i + 1
                while j < len(argv) and not argv[j].startswith("--"):
                    vals.append(argv[j])
                    j += 1
                lines.append(f"  {tok} {' '.join(vals)}")
                i = j
            else:
                lines.append(f"  {tok}")
                i += 1
        f.write(" \\\n".join(lines) + "\n")

    # 数据划分
    all_pt = sorted(glob.glob(os.path.join(args.preprocessed_root, "*.pt")))
    if not all_pt:
        raise FileNotFoundError(f"no .pt found in {args.preprocessed_root}")
    tr, va, te = split_three_ways(
        all_pt, test_ratio=args.test_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    pt_paths = {"train": tr, "val": va, "test": te, "all": all_pt}[args.split]
    if args.n > 0:
        pt_paths = pt_paths[: args.n]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        args.model,
        in_channels=1,
        out_channels=args.num_classes,
        img_size=tuple(args.patch),
    ).to(device)
    load_ckpt(args.ckpt, model, optimizer=None, map_location=device)
    model.eval()

    print(f"[eval] ckpt={args.ckpt}")
    print(
        f"[eval] model={args.model}  num_classes={args.num_classes}  patch={args.patch}"
    )
    print(f"[eval] split={args.split}  n_cases={len(pt_paths)}  device={device}")

    liver_metrics_list: List[Dict] = []
    tumor_metrics_list: List[Dict] = []
    rows: List[Dict] = []
    time_start = time.time()

    with torch.no_grad():
        for case_idx, pt_path in enumerate(pt_paths, start=1):
            case_name = os.path.basename(pt_path).replace(".pt", "")
            data = torch.load(
                pt_path, map_location="cpu", weights_only=False, mmap=True
            )
            image = data["image"].float()  # [1,D,H,W]
            label = data.get("label", None)  # [1,D,H,W] or None

            x = image.unsqueeze(0).to(device)  # [1,1,D,H,W]

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")
            ):
                if args.tta:
                    logits = tta_infer(
                        x, tuple(args.patch), args.sw_batch_size, model, args.overlap
                    )
                else:
                    logits = sliding_window_inference(
                        x, tuple(args.patch), args.sw_batch_size, model, args.overlap
                    )

            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits = torch.as_tensor(logits).float()

            pred = torch.argmax(logits, dim=1)[0].cpu()  # [D,H,W]

            # 肝脏 mask（类别1），保留最大连通域
            liver_mask = filter_largest_component(pred == 1)

            # 肿瘤 mask（类别2，仅在 num_classes>=3 时有意义）
            if args.num_classes >= 3:
                raw_tumor = pred == 2
                tumor_mask = raw_tumor & liver_mask  # 约束在肝脏内

                # 去除小连通域
                labeled, num = ndi.label(tumor_mask.cpu().numpy())
                sizes = ndi.sum(tumor_mask.cpu().numpy(), labeled, range(1, num + 1))
                clean = torch.zeros_like(tumor_mask)
                for comp_idx, s in enumerate(sizes):
                    if s > args.min_tumor_size:
                        clean[labeled == (comp_idx + 1)] = 1
                tumor_mask = clean.bool()
            else:
                tumor_mask = torch.zeros_like(liver_mask)

            # 最终预测：0=背景，1=肝脏，2=肿瘤
            final_pred = torch.zeros_like(pred, dtype=torch.long)
            final_pred[liver_mask] = 1
            final_pred[tumor_mask] = 2

            row: Dict = {
                "case_name": case_name,
                "source_pt": os.path.basename(pt_path),
                "pred_liver_voxels": int(liver_mask.sum().item()),
                "pred_tumor_voxels": int(tumor_mask.sum().item()),
            }

            if label is not None:
                gt = label[0].long()  # [D,H,W]
                gt_liver = gt > 0
                gt_tumor = gt == 2

                liver_m = compute_metrics(final_pred > 0, gt_liver)
                tumor_m = compute_metrics(final_pred == 2, gt_tumor)

                row["liver_dice"] = round(liver_m["Dice"], 4)
                row["tumor_dice"] = round(tumor_m["Dice"], 4)
                row["tumor_jaccard"] = round(tumor_m["Jaccard"], 4)
                row["tumor_recall"] = round(tumor_m["Recall"], 4)
                row["tumor_precision"] = round(tumor_m["Precision"], 4)
                row["tumor_FDR"] = round(tumor_m["FDR"], 4)

                liver_metrics_list.append(liver_m)
                tumor_metrics_list.append(tumor_m)

            rows.append(row)

            if vis_dir is not None and case_idx <= args.vis_n:
                save_case_visualization(
                    save_path=os.path.join(vis_dir, f"{case_name}.png"),
                    image=image,
                    label=label,
                    pred1=liver_mask.long(),
                    tumor_full=tumor_mask.long(),
                    final_pred=final_pred,
                    case_name=case_name,
                )

            if pred_dir is not None:
                torch.save(
                    {
                        "image": image,
                        "label": label,
                        "pred": final_pred.unsqueeze(0).long(),
                        "meta": row,
                    },
                    os.path.join(pred_dir, f"{case_name}_pred.pt"),
                )

            msg = f"[{case_idx}/{len(pt_paths)}] {case_name}"
            if "liver_dice" in row:
                msg += f"  liver={row['liver_dice']:.4f}  tumor={row['tumor_dice']:.4f}"
            print(msg)

    elapsed_hours = (time.time() - time_start) / 3600.0

    # per-case CSV
    csv_path = os.path.join(workdir, "per_case.csv")
    fieldnames = [
        "case_name",
        "source_pt",
        "pred_liver_voxels",
        "pred_tumor_voxels",
        "liver_dice",
        "tumor_dice",
        "tumor_jaccard",
        "tumor_recall",
        "tumor_precision",
        "tumor_FDR",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # metrics.json
    metrics = {
        "split": args.split,
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "n_cases": len(rows),
        "device": device,
        "elapsed_hours": round(elapsed_hours, 3),
        "liver": summarize_metrics_list(liver_metrics_list, ["Dice"]),
        "tumor": summarize_metrics_list(
            tumor_metrics_list, ["Dice", "Jaccard", "Recall", "FDR", "FNR", "Precision"]
        ),
    }
    with open(os.path.join(workdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # report.txt
    with open(os.path.join(workdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("Evaluation Report\n")
        f.write("=================\n")
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"workdir: {workdir}\n")
        f.write(f"split: {metrics['split']}\n")
        f.write(f"seed: {metrics['seed']}\n")
        f.write(f"val_ratio: {metrics['val_ratio']}\n")
        f.write(f"test_ratio: {metrics['test_ratio']}\n")
        f.write(f"n_cases: {metrics['n_cases']}\n")
        f.write(f"device: {metrics['device']}\n")
        f.write(f"elapsed_hours: {metrics['elapsed_hours']}\n\n")
        for organ, organ_key in [("Liver", "liver"), ("Tumor", "tumor")]:
            f.write(f"{organ}\n")
            for metric_name, summary in metrics[organ_key].items():
                f.write(f"  {metric_name}\n")
                f.write(f"    mean: {summary['mean']}\n")
                f.write(f"     std: {summary['std']}\n")
                f.write(f"     min: {summary['min']}\n")
                f.write(f"     max: {summary['max']}\n")
            f.write("\n")

    print("\n===== Final Metrics =====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {workdir}")


if __name__ == "__main__":
    main()
