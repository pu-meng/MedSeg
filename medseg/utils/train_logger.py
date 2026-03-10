import os
import csv
import math
from datetime import datetime


class TrainLogger:
    """
    同时写 log.csv 和 log.txt,train.py 只需调用 .log() 一行.
    """

    def __init__(self, workdir: str):
        self.csv_path = os.path.join(workdir, "log.csv")
        self.txt_path = os.path.join(workdir, "log.txt")
        self.fieldnames = [
            "time", "epoch", "train_loss",
            "val_liver_dice", "val_tumor_dice",
            "best_score", "lr",
        ]
        # 写 csv 表头
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
        # 写 txt 表头
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write(f"{'time':<14} {'epoch':>5} {'loss':>7} {'liver':>7} {'tumor':>7} {'best':>7} {'lr':>9}\n")
            f.write("-" * 60 + "\n")

    def log(self, epoch, train_loss, val_liver, val_tumor, best, lr):
        """
        val_liver / val_tumor 传 float('nan') 表示本轮没有验证
        """
        now = datetime.now().strftime("%m-%d %H:%M")
        has_val = not math.isnan(val_liver)

        # --- 写 csv ---
        row = {
            "time":          now,
            "epoch":         epoch,
            "train_loss":    round(train_loss, 4),
            "val_liver_dice": round(val_liver, 4) if has_val else "",
            "val_tumor_dice": round(val_tumor, 4) if has_val else "",
            "best_score":    round(best, 4),
            "lr":            round(lr, 6),
        }
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)

        # --- 写 txt ---
        liver_s = f"{val_liver:.4f}" if has_val else "   -  "
        tumor_s = f"{val_tumor:.4f}" if has_val else "   -  "
        line = (f"{now:<14} {epoch:>5} {train_loss:>7.4f} "
                f"{liver_s:>7} {tumor_s:>7} {best:>7.4f} {lr:>9.2e}\n")
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(line)