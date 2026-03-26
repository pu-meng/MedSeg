# medseg/utils/io_utils.py
"""
io_utils.py

from typing import Any, Dict
这里的typing是Python类型标注模块,用来给函数参数添加类型说明

def save_json(obj: Dict[str, Any], out_dir: str, name: str) -> str:
这里的obj必须是一个字典,键为字符串,值为任意类型
out_dir必须是一个字符串,表示输出目录

sys.argv是一个列表,包含了命令行参数

os.environ.get是读取环境变量

a.startswith("-")是判断字符串是否以-开头

a.lstrip("-")去掉字符串开头的-


环境变量=操作系统保存的一组变量
给程序提供运行环境信息,程序可以随时读取
例如:PATH,HOME,CUDA_VISIBLE_DEVICES,USER
1.可以临时设置环境变量
CUDA_VISIBLE_DEVICES=0 python train.py
只对这个命令生效

2. export CUDA_VISIBLE_DEVICES=0
直到关闭终端,这个变量会一直存在

3. ~/.bashrc 写入配置文件
加入 export CUDA_VISIBLE_DEVICES=0
每次打开终端,这个变量就会自动设置

CUDA内存策略
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
减少显存碎片

PYTHONPATH=/home/user/project
这个是Python解释器搜索模块的路径


export的作用是,||将后面的变量传递给后面的命令
"""


import os
import json
import sys
from typing import Any, Dict


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_cmd(out_dir: str, filename: str = "cmd.txt") -> str:
    """
    按 flag(--xxx)分组记录命令行参数,每个flag及其所有值一行。
    正确处理 nargs=3 的参数，如 --early_ratios 0 1 0。
    """
    path = os.path.join(out_dir, filename)
    args = sys.argv[1:]
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    cuda_prefix = f"CUDA_VISIBLE_DEVICES={cuda} " if cuda is not None else ""

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
        script = sys.argv[0]
        f.write(f"{cuda_prefix}python {script} \\\n")
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
