import os, csv
from datetime import datetime


class CSVLogger:
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
