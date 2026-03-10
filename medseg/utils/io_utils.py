# medseg/utils/io_utils.py
import os
import json
import sys
from typing import Any, Dict


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_cmd(out_dir: str, filename: str = "cmd.txt") -> str:
    path = os.path.join(out_dir, filename)
    # 每个参数一行,更易读
    with open(path, "w", encoding="utf-8") as f:
        f.write(sys.argv[0] + " \\\n")  # 脚本名
        args = sys.argv[1:]
        for i in range(0, len(args), 2):  # 每对参数一行
            pair = " ".join(args[i : i + 2])
            end = " \\\n" if i + 2 < len(args) else "\n"
            f.write("  " + pair + end)
    return path


def save_json(obj: Dict[str, Any], out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def save_report(text: str, out_dir: str, filename: str = "report.txt") -> str:
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
    return path
