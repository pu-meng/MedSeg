import os
import time
import json
import argparse
import torch

from medseg.utils.io_utils import save_cmd, save_json, save_report
from medseg.data.msd import load_msd_dataset
from medseg.data.build_loader import build_loaders
from medseg.models.build_model import build_model
from medseg.engine.train_eval import validate_sliding_window
from medseg.utils.ckpt import load_ckpt
from medseg.utils.warnings import setup_warnings
from medseg.tasks import get_task

from medseg.data.build_loader import build_loaders_offline
from medseg.data.dataset_offline import load_pt_paths
from medseg.utils.train_utils import split_three_ways

setup_warnings()

# 你机器上的默认路径（自用省事）
DEFAULT_DATA_ROOT = "/home/pumengyu/Task03_Liver_pt"
DEFAULT_EXP_ROOT = "/home/pumengyu/experiments"


def pick_arg(cli_value, train_cfg, key, default=None, cast_fn=None):
    """
    优先级：
    1) 命令行传入
    2) train config
    3) 硬编码默认值
    """
    if cli_value is not None:
        value = cli_value
    elif key in train_cfg and train_cfg[key] is not None:
        value = train_cfg[key]
    else:
        value = default

    if cast_fn is not None and value is not None:
        value = cast_fn(value)
    return value


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--task", type=str, default="liver", choices=["heart", "liver"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--num_classes", type=int, default=None)

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--model", type=str, default=None)

    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--test_ratio", type=float, default=None)
    p.add_argument("--patch", type=int, nargs=3, default=None)
    p.add_argument("--sw_batch_size", type=int, default=None)
    p.add_argument("--overlap", type=float, default=None)

    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--cache_rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--exp_root", type=str, default=DEFAULT_EXP_ROOT)
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument(
        "--preprocessed_root",
        type=str,
        default=None,
        help="如果传入则走离线 .pt 模式，否则走原始 .nii.gz 模式",
    )

    return p.parse_args()


def load_train_config(train_config_path):
    if os.path.isfile(train_config_path):
        with open(train_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            # 把文件里面的json文本解析成Python对象，cfg是一个dict
        print("Loaded train config:", train_config_path)
        return cfg
    return {}


def resolve_args(args, train_cfg, task_cfg):
    args.data_root = pick_arg(
        args.data_root, train_cfg, "data_root", task_cfg["data_root"], str
    )
    args.num_classes = pick_arg(
        args.num_classes, train_cfg, "num_classes", int(task_cfg["num_classes"]), int
    )

    args.val_ratio = pick_arg(args.val_ratio, train_cfg, "val_ratio", 0.2, float)
    args.test_ratio = pick_arg(args.test_ratio, train_cfg, "test_ratio", 0.1, float)
    args.seed = pick_arg(args.seed, train_cfg, "seed", 0, int)

    args.patch = pick_arg(
        args.patch, train_cfg, "patch", [96, 96, 96], lambda x: list(map(int, x))
    )
    args.model = pick_arg(args.model, train_cfg, "model", "unet3d", str)
    args.sw_batch_size = pick_arg(
        args.sw_batch_size, train_cfg, "sw_batch_size", 1, int
    )
    args.overlap = pick_arg(args.overlap, train_cfg, "overlap", 0.5, float)

    args.num_workers = pick_arg(args.num_workers, train_cfg, "num_workers", 2, int)
    args.cache_rate = pick_arg(args.cache_rate, train_cfg, "cache_rate", 0.0, float)

    if args.preprocessed_root is None:
        args.preprocessed_root = train_cfg.get("preprocessed_root", None)

    print("Resolved eval args:")
    print(f"  model={args.model}")
    print(f"  patch={args.patch}")
    print(f"  sw_batch_size={args.sw_batch_size}")
    print(f"  overlap={args.overlap}")
    print(f"  cache_rate={args.cache_rate}")
    print(f"  num_workers={args.num_workers}")
    print(f"  val_ratio={args.val_ratio}")
    print(f"  seed={args.seed}")
    print(f"  preprocessed_root={args.preprocessed_root}")
    print(f"test_ratio={args.test_ratio}")

    return args


def build_val_loader(args):
    if args.preprocessed_root:
        print(f"[离线模式] 读取 .pt 文件: {args.preprocessed_root}")

        all_pt = load_pt_paths(args.preprocessed_root)
        _, _, te_paths = split_three_ways(
            all_pt, test_ratio=args.test_ratio, val_ratio=args.val_ratio, seed=args.seed
        )

        _, test_loader = build_loaders_offline(
            [],
            te_paths,
            patch_size=tuple(args.patch),
            batch_size=1,
            num_workers=args.num_workers,
        )
        print(f"测试案例: {len(te_paths)}")
        return test_loader

    print(f"[在线模式] 读取 .nii.gz: {args.data_root}")
    items, _ = load_msd_dataset(args.data_root)
    
    _, _, te = split_three_ways(
        items, test_ratio=args.test_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    # 注意：build_loaders 需要 tr+va 两个参数，这里把 te 当 va 传进去
    _, val_loader = build_loaders(
        [],  # train 为空，只需要 val_loader
        te,
        patch_size=tuple(args.patch),
        batch_size=1,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
    )
    print(f"测试案例: {len(te)}")

    return val_loader


def main():
    args = parse_args()
    task_cfg = get_task(args.task)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.abspath(args.ckpt)
    train_run_dir = os.path.dirname(ckpt_path)  # .../train/<timestamp>
    train_ts = os.path.basename(train_run_dir)  # <timestamp>
    train_parent = os.path.basename(os.path.dirname(train_run_dir))  # train

    if train_parent != "train":
        raise ValueError(f"ckpt must be under .../train/<timestamp>/, got: {ckpt_path}")

    auto_out_dir = os.path.join(args.exp_root, args.exp_name, "eval", train_ts)
    out_dir = args.out_dir if args.out_dir is not None else auto_out_dir
    os.makedirs(out_dir, exist_ok=True)
    print("Eval out_dir:", out_dir)

    train_config_path = os.path.join(
        args.exp_root, args.exp_name, "train", train_ts, "config.json"
    )
    train_cfg = load_train_config(train_config_path)
    args = resolve_args(args, train_cfg, task_cfg)

    val_loader = build_val_loader(args)

    model = build_model(
        args.model,
        in_channels=1,
        out_channels=int(args.num_classes),
        img_size=tuple(args.patch),
    ).to(device)

    ckpt = load_ckpt(args.ckpt, model, optimizer=None, map_location=device)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    metrics = validate_sliding_window(
        model,
        val_loader,
        device,
        roi_size=tuple(args.patch),
        sw_batch_size=args.sw_batch_size,
        num_classes=int(args.num_classes),
        overlap=args.overlap,
    )

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    total_sec = t1 - t0
    n_cases = len(val_loader.dataset) if hasattr(val_loader, "dataset") else None
    sec_per_case = (total_sec / n_cases) if (n_cases and n_cases > 0) else None

    val_mean = float(metrics["mean_fg"])
    per_class = metrics["per_class"]
    class_ids = metrics["class_ids"]

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    train_epochs = train_cfg.get("epochs", "NA")

    config = {
        "task": args.task,
        "exp_name": args.exp_name,
        "exp_root": args.exp_root,
        "out_dir": out_dir,
        "train_epochs": train_epochs,
        "ckpt": args.ckpt,
        "model": args.model,
        "data_root": args.data_root,
        "num_classes": int(args.num_classes),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "seed": int(args.seed),
        "patch": list(args.patch),
        "sw_batch_size": int(args.sw_batch_size),
        "overlap": float(args.overlap),
        "num_workers": int(args.num_workers),
        "cache_rate": float(args.cache_rate),
        "device": device,
        "gpu_name": gpu_name,
        "preprocessed_root": args.preprocessed_root,
    }

    metrics_out = {
        "best_epoch": ckpt.get("epoch", "NA"),
        "n_test_cases": int(n_cases) if n_cases is not None else None,
        "total_infer_seconds": float(total_sec),
        "seconds_per_case": float(sec_per_case) if sec_per_case is not None else None,
        "val_dice_mean_foreground": float(val_mean),
        "val_dice_per_class": [float(x) for x in per_class],
        "val_dice_class_ids": [int(x) for x in class_ids],
    }

    report_lines = [
        f"epoch: {metrics_out['best_epoch']}",
        f"val_dice_mean_foreground: {metrics_out['val_dice_mean_foreground']:.6f}",
        f"val_dice_per_class: {metrics_out['val_dice_per_class']} (class_ids={metrics_out['val_dice_class_ids']})",
        f"n_test_cases: {metrics_out['n_test_cases']}",
        f"total_infer_seconds: {metrics_out['total_infer_seconds']:.2f}",
        f"seconds_per_case: {metrics_out['seconds_per_case']}",
    ]
    report = "\n".join(report_lines)
    print(report)

    save_cmd(out_dir)
    save_json(config, out_dir, "config")
    save_json(metrics_out, out_dir, "metrics")
    save_report(report, out_dir)

    print("Saved:")
    print(" -", os.path.join(out_dir, "report.txt"))
    print(" -", os.path.join(out_dir, "config.json"))
    print(" -", os.path.join(out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
