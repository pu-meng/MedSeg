import os
import sys
import time
import argparse
import torch
from datetime import datetime

from medseg.models.build_model import build_model
from medseg.utils.ckpt import save_ckpt, load_ckpt
from medseg.utils.train_logger import TrainLogger
from medseg.utils.io_utils import ensure_dir, save_json, save_report
from medseg.utils.train_utils import set_seed
from medseg.data.msd import fixed_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage2.engine import train_one_epoch, validate, build_loss
from stage2.loader import load_stage2_items, build_loaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crops_dir", type=str, required=True)
    p.add_argument("--exp_root", type=str, default="/home/pumengyu/experiments")
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--model", type=str, default="unet3d")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--patch", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--sw_batch_size", type=int, default=4)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--val_every", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def main():
    start_time = time.time()
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    workdir = os.path.join(args.exp_root, args.exp_name, "train", timestamp)
    ensure_dir(workdir)
    print("Stage2 保存路径:", workdir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

    config = vars(args)
    config.update(
        {
            "workdir": workdir,
            "device": device,
            "gpu_name": gpu_name,
            "timestamp": timestamp,
            "stage": 2,
        }
    )
    save_json(config, workdir, "config")

    # 数据
    all_items = load_stage2_items(args.crops_dir)
    tr_items, va_items = fixed_split(
        all_items, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"train={len(tr_items)}  val={len(va_items)}")

    patch_size = tuple(args.patch)
    train_loader, val_loader = build_loaders(
        tr_items,
        va_items,
        patch_size,
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
        ratios=(0.1, 0.9),
    )

    # 模型 / 优化器 / loss
    model = build_model(
        args.model, in_channels=1, out_channels=2, img_size=patch_size
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler() if (args.amp and device == "cuda") else None
    loss_fn = build_loss("dicece")

    logger = TrainLogger(workdir)
    best = -1.0
    best_epoch = -1
    start_epoch = 1

    if args.resume and os.path.exists(args.resume):
        ckpt = load_ckpt(args.resume, model, optimizer, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best = ckpt["best_metric"]
        print(f"Resume 从 epoch {ckpt['epoch']}, best={best:.4f}")

    val_tumor_dice = float("nan")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print("=" * 50)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            loss_fn,
            epoch,
            args.epochs,
        )

        did_val = epoch % args.val_every == 0
        if did_val:
            val_tumor_dice = validate(
                model,
                val_loader,
                device,
                roi_size=patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.overlap,
            )
            if val_tumor_dice > best:
                best, best_epoch = val_tumor_dice, epoch
                save_ckpt(
                    os.path.join(workdir, "best.pt"), model, optimizer, epoch, best
                )
                print(f"  ✅ 新最优 best={best:.4f}")

        save_ckpt(os.path.join(workdir, "last.pt"), model, optimizer, epoch, best)

        lr = float(optimizer.param_groups[0]["lr"])
        now = datetime.now().strftime("%m-%d %H:%M")
        logger.log(epoch, train_loss, float("nan"), val_tumor_dice, best, lr)

        dt_m = (time.time() - t0) / 60
        if did_val:
            print(
                f"[{now}][Ep {epoch:03d}] loss={train_loss:.4f}  "
                f"tumor_dice={val_tumor_dice:.4f}  best={best:.4f}  {dt_m:.1f}m"
            )
        else:
            print(f"[{now}][Ep {epoch:03d}] loss={train_loss:.4f}  {dt_m:.1f}m")

    # 结束
    elapsed = time.time() - start_time
    save_json(
        {
            "best_tumor_dice": best,
            "best_epoch": best_epoch,
            "elapsed_hours": round(elapsed / 3600, 2),
            "train_n": len(tr_items),
            "val_n": len(va_items),
        },
        workdir,
        "metrics",
    )
    report = (
        f"Stage2 训练完成\nbest_tumor_dice={best:.4f} (epoch {best_epoch})\n"
        f"耗时 {elapsed / 3600:.1f}h\n"
    )
    save_report(report, workdir)
    print(report)


if __name__ == "__main__":
    main()
