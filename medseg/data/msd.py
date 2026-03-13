import os
import glob
from typing import List, Dict, Tuple
"""
msd.py
负责读取MSD/nnUNet风格的数据集,并进行安全配对和划分.
输入: MSD/nnUNet风格的数据集路径
输出: train_items, test_items, 以及 train-val 划分

本质是在线模式的"样本索引构建器"
"""

def _sorted_nii(folder: str) -> List[str]:
    files = glob.glob(os.path.join(folder, "*.nii.gz")) + glob.glob(
        os.path.join(folder, "*.nii")
    )
    return sorted(files)


def _sid_from_path(p: str) -> str:
    """
    从文件路径提取ID,例如:
    输入: /path/to/imagesTr/Case_00001.nii.gz
    输出: Case_00001;
    os.path.basename(p) -> Case_00001.nii.gz
    .replace(".nii.gz", "") -> Case_00001
    """
    return os.path.basename(p).replace(".nii.gz", "").replace(".nii", "")


def load_msd_dataset(task_root: str) -> Tuple[List[Dict], List[Dict]]:
    """
    - MSD/nnUNet 风格数据集读取(安全配对版)
    - 返回:
      - train_items: [{"image": path, "label": path, "id": str}, ...]
      - test_items:  [{"image": path, "id": str}, ...]

    """
    imagesTr = os.path.join(task_root, "imagesTr")
    labelsTr = os.path.join(task_root, "labelsTr")

    tr_images = _sorted_nii(imagesTr)
    tr_labels = _sorted_nii(labelsTr)

    if len(tr_images) == 0 or len(tr_labels) == 0:
        raise RuntimeError(f"Empty imagesTr/labelsTr under: {task_root}")
    if len(tr_images) != len(tr_labels):
        raise RuntimeError(
            f"Mismatch: imagesTr={len(tr_images)} labelsTr={len(tr_labels)}"
        )

    # label 映射:id -> label_path
    label_map: Dict[str, str] = {}
    # 这个等价于 label_map=  {}
    #:Dict[str, str] 这个是类型注解

    for lab in tr_labels:
        sid = _sid_from_path(lab)
        if sid in label_map:
            raise RuntimeError(f"Duplicate label id found: {sid}")
        label_map[sid] = lab

    # 用 image 的 id 去查 label,确保 1-1 对齐
    train_items: List[Dict] = []
    # 告诉电脑,train_items 是一个列表,列表中的元素是字典
    for img in tr_images:
        sid = _sid_from_path(img)
        if sid not in label_map:
            raise RuntimeError(f"Label missing for {sid}")
        train_items.append({"image": img, "label": label_map[sid], "id": sid})

    # 可选:也检查 label 是否存在对应 image(更严格)
    image_ids = {x["id"] for x in train_items}
    extra_labels = [sid for sid in label_map.keys() if sid not in image_ids]
    if len(extra_labels) > 0:
        raise RuntimeError(f"Labels without images: {extra_labels[:10]}")

  

  
    return train_items


def fixed_split(items, val_ratio: float = 0.2, seed: int = 0):
    """
    可复现随机划分:
    - 用 seed 控制 shuffle
    - 不同 seed -> 不同划分(用于多次实验/更有说服力)
    """
    import random

    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_val = max(1, int(round(n * val_ratio)))
    val_items = items[:n_val]
    train_items = items[n_val:]
    return train_items, val_items

