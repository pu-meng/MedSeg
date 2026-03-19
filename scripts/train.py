"""
ratios:控制训练时候采样的比例
overlap:在推理出使用,让预测结果在边界处平滑

步长=(1-overlap)*patch长度
需要几个=(CT长度-步长)/步长+1
"""

import os
import argparse

import time
import torch
from datetime import datetime
import warnings
from medseg.utils.warnings import setup_warnings


from medseg.utils.train_utils import (
    set_seed,
    get_stage_ratios,
    build_metrics,
    build_report,
    load_data,
    build_loaders_auto,
)
from medseg.utils.ckpt import load_ckpt


from medseg.models.build_model import build_model
from medseg.engine.train_eval import train_one_epoch_multiclass, validate_sliding_window

from medseg.utils.ckpt import save_ckpt
from medseg.tasks import get_task

from medseg.utils.train_logger import TrainLogger
from pathlib import Path

# ✅ 新增:统一输出工具
from medseg.utils.io_utils import ensure_dir, save_cmd, save_json, save_report

setup_warnings()


warnings.filterwarnings("ignore", message="no available indices of class")
DEFAULT_DATA_ROOT = "/home/pumengyu/Task03_Liver"
DEFAULT_EXP_ROOT = "/home/pumengyu/experiments"


def short(path, keep=3):
    """
    path:输入路径
    keep:保留几段路径

    Linux:/home/pumengyu/Desktop/medseg
    所以要先统一成/
    parts=['','home','user'];
    Path(path).parts=['C:','Users','pumengyu'];
    from pathlib import Path
    Path("C:/Users/pumengyu/Desktop/medseg").parts=['C:','Users','pumengyu']
    Path的返回对象是Path对象,自动适配Windows/Linux,
    可以直接.name和.parent,.exists()等操作;
    parts的意思:把路径按/分割成列表,返回元组
    Path对象有,.suffix返回后缀,.parent返回父目录等


    """

    parts = Path(path).parts
    return "/".join(parts[-keep:])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="liver", choices=["heart", "liver"])
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="数据集,如果为None,则使用默认路径",
    )
    p.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="类别数,如果为None,则使用默认值",
    )
    p.add_argument(
        "--train_n",
        type=int,
        default=0,
        help="仅使用前N个训练样本进行快速调试(0=全部)",
    )
    p.add_argument(
        "--val_n",
        type=int,
        default=0,
        help="仅使用前N个验证样本进行快速调试(0=全部)",
    )

    p.add_argument("--exp_root", type=str, default=DEFAULT_EXP_ROOT)
    p.add_argument(
        "--exp_name",
        type=str,
        default="debug",
        help="实验名称,会保存在 exp_root/exp_name/ 下",
    )

    p.add_argument(
        "--model",
        type=str,
        default="unet3d",
        help="模型名称,如 unet, unetr, attention_unet(attunet)",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--patch", type=int, nargs=3, default=[144, 144, 144])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_rate", type=float, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--sw_batch_size", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--loss",
        type=str,
        default="dicece",
        choices=["dicece", "dicefocal", "tversky", "focaltversky"],
    )
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument(
        "--preprocessed_root",
        type=str,
        default=None,
        help="离线预处理 .pt 文件目录.传入则走离线流程,不传则走原来的 .nii.gz 流程",
    )
    p.add_argument(
        "--overlap", type=float, default=0.5, help="滑窗推理重叠率,建议0.25`"
    )
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="DataLoader prefetch_factor,建议根据内存调整,先用默认值",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="离线数据集重复次数,增加epoch内迭代次数,建议根据训练时长调整,先用默认值1",
    )
    p.add_argument("--resume", type=str, default=None, help="Checkpoint路径")
    p.add_argument(
        "--early_ratios",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="前半段采样比例 背景/前景,前景包括肝脏和肿瘤",
    )
    p.add_argument(
        "--late_ratios",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="后半段采样比例 背景/前景,前景包括肝脏和肿瘤",
    )
    p.add_argument("--test_ratio", type=float, default=0.1)

    p.add_argument(
        "--val_patch",
        type=int,
        nargs=3,
        default=None,
        help="验证滑窗roi_size；不传则默认等于patch",
    )
    p.add_argument(
        "--merge_label12_to1",
        action="store_true",
        help="将label中的1和2合并为1，用于liver-only二分类训练",
    )

    return p.parse_args()


def main():
    """
    args=parse_args()等价于,先运行这个函数,再运行后面的代码
    """
    start_time = time.time()
    args = parse_args()
    if args.val_patch is None:
        args.val_patch = args.patch
    set_seed(args.seed)

    task_cfg = get_task(args.task)
    # 命令行参数>>配置文件>默认值
    if args.data_root is None:
        args.data_root = task_cfg.get("data_root", DEFAULT_DATA_ROOT)
    if args.num_classes is None:
        if "num_classes" not in task_cfg:
            raise ValueError(
                f"task_cfg for '{args.task}' 未指定 num_classes，请在任务配置中明确设置"
            )
        args.num_classes = int(task_cfg["num_classes"])
    # ✅ 不改时间戳格式
    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    workdir = os.path.join(args.exp_root, args.exp_name, "train", timestamp)
    ensure_dir(workdir)
    print("训练保存路径:", workdir)

    # ✅ 统一落盘:cmd + config(第1时间写,保证可复现)
    save_cmd(workdir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

    # config.json:只放配置 + 环境(不放指标)
    config = {
        "task": args.task,
        "exp_root": args.exp_root,
        "exp_name": args.exp_name,
        "workdir": workdir,
        "data_root": args.data_root,
        "num_classes": int(args.num_classes),
        "train_n": int(args.train_n),
        "val_n": int(args.val_n),
        "model": args.model,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "patch": list(args.patch),
        "num_workers": int(args.num_workers),
        "cache_rate": float(args.cache_rate),
        "amp": bool(args.amp),
        "sw_batch_size": int(args.sw_batch_size),
        "seed": int(args.seed),
        "device": device,
        "gpu_name": gpu_name,
        "val_every": int(args.val_every),
        "timestamp": timestamp,
        "loss": args.loss,
        "overlap": float(args.overlap),
        "prefetch_factor": int(args.prefetch_factor),
        "preprocessed_root": args.preprocessed_root,  # 方便你对齐目录
        "repeats": int(args.repeats),
        "resume": args.resume,
        "early_ratios": list(args.early_ratios),
        "late_ratios": list(args.late_ratios),
        "optimizer": "SGD",
        "momentum": 0.99,
        "weight_decay": 3e-5,
        "lr_scheduler": "poly_0.9",
        "merge_label12_to1": bool(args.merge_label12_to1),
    }
    save_json(config, workdir, "config")

    # 数据加载与 split
    # 数据加载:离线 .pt 或原来的 .nii.gz
    # 这个地方用到val_ratio，所以不能放在load_data之前
    tr, va, te, use_offline = load_data(args)

    # model/optim
    model = build_model(
        args.model,
        in_channels=1,
        out_channels=int(args.num_classes),
        img_size=tuple(args.patch),
    ).to(device)

    optim = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.99, nesterov=True, weight_decay=3e-5
    )
    # optim 下面加
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9
    )
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device == "cuda") else None

    logger = TrainLogger(workdir)

    best = -1.0

    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = load_ckpt(args.resume, model, optim, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best = ckpt["best_metric"]  # ← 注意是 best_metric
        print(f"resume 从 epoch {ckpt['epoch']} 继续, best={best:.4f}")
        for _ in range(ckpt["epoch"]):
            scheduler.step()

    best_epoch = -1
    best_c1 = None
    best_c2 = None

    val_c1 = float("nan")
    val_c2 = float("nan")

    init_ratios = get_stage_ratios(
        1, args.epochs, tuple(args.early_ratios), tuple(args.late_ratios)
    )

    train_loader, val_loader = build_loaders_auto(
        args, tr, va, use_offline, train_ratios=init_ratios
    )
    current_train_ratios = init_ratios

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print("=" * 50)
        print(f"Epoch {epoch}/{args.epochs}")

        # ✅ 分阶段 ratios:每个 epoch 动态创建 train_loader
        new_train_ratios = get_stage_ratios(
            epoch, args.epochs, tuple(args.early_ratios), tuple(args.late_ratios)
        )
        print(f"[epoch {epoch}] crop ratios={new_train_ratios}")
        if new_train_ratios != current_train_ratios:
            current_train_ratios = new_train_ratios
            print(
                f"更新采样比例为 {current_train_ratios},重新缓存+重新创建 train_loader(val_loader 不变)"
            )
            train_loader, _ = build_loaders_auto(
                args, tr, va, use_offline, current_train_ratios
            )
        train_loss = train_one_epoch_multiclass(
            model,
            train_loader,
            optim,
            device,
            scaler=scaler,
            loss_type=args.loss,
            epoch=epoch,
            epochs=args.epochs,
        )
        print(f"Epoch {epoch}  loss_type: {args.loss}  train_loss: {train_loss:.4f}")

        did_val = False
        score = float("nan")
        if epoch % args.val_every == 0:  # 之后建议做成 args.val_every
            did_val = True
            metrics = validate_sliding_window(
                model,
                val_loader,
                device,
                roi_size=tuple(args.val_patch),
                sw_batch_size=int(args.sw_batch_size),
                num_classes=int(args.num_classes),
                return_per_class=True,
                overlap=args.overlap,
            )

            per = metrics["per_class"]
            val_c1 = float(per[0]) if len(per) > 0 else float("nan")
            val_c2 = float(per[1]) if len(per) > 1 else float("nan")
            score = val_c1

            if score > best:
                best = score
                best_epoch = epoch
                best_c1, best_c2 = val_c1, val_c2
                save_ckpt(os.path.join(workdir, "best.pt"), model, optim, epoch, best)

        # last.pt:固定频率保存,和 best 无关

        save_ckpt(os.path.join(workdir, "last.pt"), model, optim, epoch, best)
        scheduler.step()

        # 日志:建议每个 epoch 都写,至少能看训练是否“在动”
        lr = float(optim.param_groups[0]["lr"])
        now = datetime.now().strftime("%m-%d-%H-%M")

        logger.log(epoch, train_loss, val_c1, val_c2, best, lr)

        dt = time.time() - t0
        dt_m = dt / 60
        if did_val:
            if int(args.num_classes) == 2:
                print(
                    f"[{now}][Epoch {epoch:03d}] loss={train_loss:.4f}  "
                    f"liver_dice={val_c1:.4f} best={best:.4f}  time={dt_m:.1f}m"
                )
            else:
                print(
                    f"[{now}][Epoch {epoch:03d}] loss={train_loss:.4f}  "
                    f"liver_dice={val_c1:.4f} tumor_dice={val_c2:.4f} best={best:.4f}  time={dt_m:.1f}m"
                )
        else:
            print(f"[{now}][Epoch {epoch:03d}] loss={train_loss:.4f}  time={dt_m:.1f}m")

    # ✅ 训练结束:统一写 metrics.json + report.txt(只写指标,不重复配置)

    metrics_out = build_metrics(
        best,
        best_epoch,
        best_c1,
        best_c2,
        time.time() - start_time,
        args.epochs,
        len(tr),
        len(va),
    )
    report = build_report(metrics_out)
    save_json(metrics_out, workdir, "metrics")
    save_report(report, workdir)
    # 训练结束强制保存一个最终 best(如果从未验证过,就用 last 当 best)
    final_best_path = os.path.join(workdir, "best.pt")
    if best_epoch < 0:  # 表示从未验证过
        save_ckpt(final_best_path, model, optim, args.epochs, best)
        print("从未验证过,用 last 当 best", final_best_path)

    print(report)

    print("保存路径:", workdir)


if __name__ == "__main__":
    main()
