import os
import glob
import torch
from torch.utils.data import Dataset
import warnings

from monai.transforms import RandCropByLabelClassesd 
"""
dataset_offline.py
负责离线.pt数据的读取和划分,
输入: .pt 文件路径列表
事故出:可供DataLoader使用的offlineDataset
"""


class OfflineDataset(Dataset):
    """
    读取 preprocess_offline.py 生成的 .pt 文件.

    每个 .pt 文件格式:
        {
            "image": torch.float32 tensor [1, D, H, W],
            "label": torch.int64   tensor [1, D, H, W],
        }
    case_idx = idx % len(self.pt_paths)
    这行代码类似len(self.pt_paths)=100,那么idx=190,case_idx=90;
    这行是为了让dataset可以重复访问同一批pt文件.

    with是最重要的Python语法,它用于简化代码,提高代码的可读性.
    with something():
        do something
    进入某个环境,执行代码,退出环境
    self.pt_paths[case_idx]是取出第case_idx个.pt文件路径
    map_location="cpu"强制把数据加载到CPU
    mmap=True,不要把整个文件都读入内存,只在需要时候读取

    python对象->torch.save()->.pt文件
    .pt文件->torch.load()->python对象

    对象=一块带类型的数据
    python对象=一块带类型的数据+类型信息
    对象=数据+类型信息+能做的操作
     
     Dataset应该只返回一个sample,允许返回list,但是最多只能返回一个sample
     不然会带来混乱.

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
        """
        在transform里面找到RandCropByLabelClassesd,
        然后修改它的ratios参数
        这里的self.transform是Compose
        Compose([
        LoadImaged(...),
        Orientationd(...),
        Spacingd(...),
        ])
        self.transform.transforms一个列表
        [LoadImaged(...),Orientationd(...),Spacingd(...),]
        for t in self.transform.transforms:的意思是把pipline的每个transform取出来
        这个return是因为self.transform只有一个RandCropByLabelClassesd
        所以return一次就结束了

        pipeline是管道的意思,把数据从输入到输出,经过一系列的transform
        


        """
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