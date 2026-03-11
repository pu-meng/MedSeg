# medseg/utils/io_utils.py
import os
import json
import sys
from typing import Any, Dict


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_cmd(out_dir: str, filename: str = "cmd.txt") -> str:
    """
    按 flag(--xxx)分组记录命令行参数，每个flag及其所有值一行。
    正确处理 nargs=3 的参数，如 --early_ratios 0 1 0。
    """
    path = os.path.join(out_dir, filename)
    args = sys.argv[1:]

    # 按 -- 开头分组：每遇到新flag就开一个新组
    groups = []
    cur = []
    for a in args:
        if a.startswith("-") and not a.lstrip("-").replace(".", "").isdigit():
            if cur:
                groups.append(cur)
            cur = [a]
        else:
            cur.append(a)
    if cur:
        groups.append(cur)

    with open(path, "w", encoding="utf-8") as f:
        f.write(sys.argv[0] + " \\\n")
        for i, grp in enumerate(groups):
            line = " ".join(grp)
            end = " \\\n" if i < len(groups) - 1 else "\n"
            f.write("  " + line + end)
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
