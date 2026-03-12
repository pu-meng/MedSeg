import os
import glob
import torch
from torch.utils.data import Dataset
import warnings

from monai.transforms import RandCropByLabelClassesd 
class OfflineDataset(Dataset):
    """
    读取 preprocess_offline.py 生成的 .pt 文件.

    每个 .pt 文件格式:
        {
            "image": torch.float32 tensor [1, D, H, W],
            "label": torch.int64   tensor [1, D, H, W],
        }
    """

    def __init__(self, pt_paths: list, transform=None, repeats=1):
        self.pt_paths = pt_paths
        self.transform = transform
        self.repeats = repeats

        if len(self.pt_paths) == 0:
            raise ValueError("pt_paths 为空,OfflineDataset 无法构建.")

    def __len__(self):
        return len(self.pt_paths) * self.repeats

    def __getitem__(self, idx):
        case_idx = idx % len(self.pt_paths)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data = torch.load(
                self.pt_paths[case_idx],
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )

        if self.transform is not None:
            data = self.transform(data)

        if isinstance(data, list):
            if len(data) == 1:
                data = data[0]
            else:
                raise RuntimeError(
                    f"数据变成了 list,但长度不为 1: {len(data)},"
                    f"说明 transform 返回了多个 sample,请检查 num_samples 设置."
                )

        return data
    def set_ratios(self, ratios):
        """动态修改采样比例,不重建 loader"""
        for t in self.transform.transforms:
            if isinstance(t, RandCropByLabelClassesd):
                t.ratios = list(ratios)
                return
        raise RuntimeError("找不到 RandCropByLabelClassesd")


def load_pt_paths(preprocessed_dir: str, n: int = 0) -> list:
    paths = sorted(glob.glob(os.path.join(preprocessed_dir, "*.pt")))
    if len(paths) == 0:
        raise FileNotFoundError(f"没有找到 .pt 文件: {preprocessed_dir}")
    if n and n > 0:
        paths = paths[:n]
    return paths


def split_pt_paths(pt_paths: list, val_ratio: float = 0.2, seed: int = 0):
    import random

    rng = random.Random(seed)
    paths = pt_paths[:]
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * val_ratio))
    va = paths[:n_val]
    tr = paths[n_val:]
    return tr, va


def split_three_ways(pt_paths: list, test_ratio: float = 0.1, 
                     val_ratio: float = 0.2, seed: int = 0):
    import random
    rng = random.Random(seed)
    paths = pt_paths[:]
    rng.shuffle(paths)
    
    n_test = max(1, int(len(paths) * test_ratio))
    n_val  = max(1, int(len(paths) * val_ratio))
    
    te    = paths[-n_test:]            # 从末尾取 test
    tr_va = paths[:-n_test]            # 剩余
    va    = tr_va[-n_val:]             # 再从末尾取 val
    tr    = tr_va[:-n_val]             # 剩余就是 train
    
    return tr, va, te