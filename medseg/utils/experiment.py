# medseg/utils/experiment.py
"""
Train script for 3D medical image segmentation (MSD).

Features:
- Train with DiceCELoss
- Validate with sliding window inference
- Automatic timestamped workdir
- Save best checkpoint
- Save experiment metadata

Usage:
    python -m scripts.train \
        --data_root PATH \
        --workdir PATH \
        --model unet3d \
        --epochs 100 \
        --amp
"""
import os
import sys
import json
import torch
from datetime import datetime
from pathlib import Path


def save_run_metadata(workdir, args):
    """
    保存:
    - 运行命令
    - 配置参数
    - 运行环境信息
    """
    assert isinstance(
        workdir, (str, Path)
    ), f"workdir should be a path, got {type(workdir)}"
    os.makedirs(workdir, exist_ok=True)

    # 1️⃣ 保存命令行
    with open(os.path.join(workdir, "cmd.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # 2️⃣ 保存参数
    with open(os.path.join(workdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 3️⃣ 保存运行信息
    with open(os.path.join(workdir, "run_info.txt"), "w") as f:
        f.write(f"time: {datetime.now().isoformat()}\n")
        f.write(f"workdir: {workdir}\n")
        f.write(f"cuda_available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"gpu_name: {torch.cuda.get_device_name(0)}\n")
