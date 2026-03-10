import os
import csv
import json
import argparse
import math

def mean_std(xs):
    if len(xs) == 0:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)  # sample variance
    return m, math.sqrt(var)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="directory containing log.csv")
    ap.add_argument("--tail", type=int, default=10, help="use last N epochs for mean/std")
    args = ap.parse_args()

    csv_path = os.path.join(args.run_dir, "log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 兼容字符串 -> float/int
            rows.append({
                "epoch": int(r["epoch"]),
                "train_loss": float(r["train_loss"]),
                "val_dice": float(r["val_dice"]),
                "best_dice": float(r["best_dice"]),
                "lr": float(r["lr"]),
                "time": r.get("time", "")
            })

    rows.sort(key=lambda x: x["epoch"])
    val_dices = [r["val_dice"] for r in rows]
    losses = [r["train_loss"] for r in rows]

    best_epoch = max(rows, key=lambda r: r["val_dice"])["epoch"]
    best_val = max(val_dices)
    last_val = rows[-1]["val_dice"]
    last_loss = rows[-1]["train_loss"]

    tail = max(1, min(args.tail, len(rows)))
    tail_vals = val_dices[-tail:]
    tail_losses = losses[-tail:]
    m_dice, s_dice = mean_std(tail_vals)
    m_loss, s_loss = mean_std(tail_losses)

    summary = {
        "run_dir": os.path.abspath(args.run_dir),
        "num_epochs": len(rows),
        "best": {
            "epoch": best_epoch,
            "val_dice": best_val
        },
        "last": {
            "epoch": rows[-1]["epoch"],
            "train_loss": last_loss,
            "val_dice": last_val
        },
        "tail_window": {
            "N": tail,
            "val_dice_mean": m_dice,
            "val_dice_std": s_dice,
            "train_loss_mean": m_loss,
            "train_loss_std": s_loss
        }
    }

    out_json = os.path.join(args.run_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[SUMMARY] {args.run_dir}")
    print(f"  epochs: {summary['num_epochs']}")
    print(f"  best : epoch={best_epoch}  val_dice={best_val:.4f}")
    print(f"  last : epoch={summary['last']['epoch']}  val_dice={last_val:.4f}  loss={last_loss:.4f}")
    print(f"  tail{tail}: val_dice={m_dice:.4f}±{s_dice:.4f}  loss={m_loss:.4f}±{s_loss:.4f}")
    print(f"  saved: {out_json}")

if __name__ == "__main__":
    main()