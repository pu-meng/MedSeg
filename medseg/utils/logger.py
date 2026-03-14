import os
import csv
from datetime import datetime


class CSVLogger:
    """
        DictWriter=用字典写CSV,
        比如:
        row = {
        "epoch":1,
        "loss":0.4
    }
    csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
    写入:
    epoch,loss
    1,     0.4

    CSV可以直接变成表格
    import pandas as pd
    df = pd.read_csv("log.csv")
    做完之后可以画图
    训练指标用csv是非常有必要的
    """

    def __init__(self, csv_path: str, fieldnames):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path
        self.fieldnames = list(fieldnames)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log(self, row: dict):
        row = dict(row)
        row["time"] = datetime.now().strftime("%Y-%m-%d %H")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
