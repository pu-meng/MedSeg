import os
import glob
import torch
from torch.utils.data import Dataset
import warnings

from monai.transforms.croppad.dictionary import RandCropByLabelClassesd
from typing import Optional
from monai.transforms.compose import Compose

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

    def __init__(
        self,
        pt_paths: list,
        transform: Optional[Compose] = None,
        repeats=1,
        merge_label12_to1=False,
    ):
        self.pt_paths = pt_paths
        self.transform = transform
        self.repeats = repeats
        self.merge_label12_to1 = merge_label12_to1

        if len(self.pt_paths) == 0:
            raise ValueError("pt_paths 为空,OfflineDataset 无法构建.")

    def __len__(self):
        """
        dataset=OfflineDataset(pt_paths,repeats=6)
        len(dataset)=len(pt_paths)*repeats
        这里的len是自定义的len,规则是len作用在谁上,就调用谁的len
        len(dataset)这里的dataset是OfflineDataset的实例,所以调用OfflineDataset的len
        len(self.pt_paths):pt_paths是一个list,所以调用list的len
        """
        return len(self.pt_paths) * self.repeats

    def __getitem__(self, idx):
        """
        self.pt_paths是路径列表
        """
        case_idx = idx % len(self.pt_paths)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data = torch.load(
                self.pt_paths[case_idx],
                map_location="cpu", #将保存至任意设备上的张量,加载时强制映射到CPU内存上
                weights_only=False,#True:只允许加载张量数据,禁止执行任意Python对象的反序列化
                #weights_only=True,允许加载任何的Python对象(自定义类,dict,list等)
                mmap=True, #文件内容,不会立即全部读入RAM,而是按需从磁盘读取(操作系统级别的懒惰加载)
            )

        if self.merge_label12_to1:
            if "label" not in data:
                raise KeyError(f"数据中没有 'label' 键: {self.pt_paths[case_idx]}")
            data["label"] = (data["label"] > 0).long()  # 把标签1和2都变成1,背景是0
#.long()把数据类型转换为long,损失函数如(CrossEntropyLoss)要求标签是long类型,bool类型会报错

        if self.transform is not None:
            data = self.transform(data)
#MONAI的transform输出只有两种情况:
#dict或者list[dict],下面的代码主要排除list[dict1,dict2,...]这种情况``
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
        if self.transform is None:
            raise RuntimeError("transform 为 None,无法设置 ratios.")
        if not hasattr(self.transform, "transforms"):
            #hasattr的是has attribute的意思,判断有没有属性
            raise RuntimeError("transform 不是 Compose,无法设置 ratios.")

        for t in self.transform.transforms:
            if isinstance(t, RandCropByLabelClassesd):
                t.ratios = list(ratios)  # type:ignore
                return
        raise RuntimeError("找不到 RandCropByLabelClassesd")
#getattr(obj,"name")等价于obj.name
#getattr(obj,"name",default)等价于obj.name if hasattr(obj,"name") else default
def load_pt_paths(preprocessed_dir: str, n: int = 0) -> list:
    """
    返回的是路径列表,
    paths是列表,sorted()永远返回list,glob 是Unix shell的通配符匹配的缩写(global match)
    glob.glob(pattern)返回所有匹配的pattern的文件路径,输出是list,

    比如:
    path=glob.glob("/home/pumengyu/**/*.mp4",recursive=True)
    **表示匹配任意层级的子目录
    recursive=True,表示递归匹配,即匹配所有子目录中的文件,必须加,不然,**不生效
    输出是所有匹配到mp4的完整路径列表
    from pathlib import Path
    paths=list(Path("/home/pumengyu").rglob("*.mp4"))
    rglob等价于递归glob,也就是glob("**/*.mp4",recursive=True)
    """
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


def split_three_ways(
    pt_paths: list, test_ratio: float = 0.1, val_ratio: float = 0.2, seed: int = 0
):
    """
    split_three_ways函数的作用是将pt_paths列表中的数据按照test_ratio和val_ratio的比例分割成三个部分,分别是训练集、验证集和测试集。
    这个函数应该是two-stage的两个阶段的唯一的数据分割函数,确保分割比例一致
    """
    import random

    rng = random.Random(seed)
    paths = pt_paths[:]
    rng.shuffle(paths)

    n_test = max(1, int(len(paths) * test_ratio))
    n_val = max(1, int(len(paths) * val_ratio))

    te = paths[-n_test:]  # 从末尾取 test
    tr_va = paths[:-n_test]  # 剩余
    va = tr_va[-n_val:]  # 再从末尾取 val
    tr = tr_va[:-n_val]  # 剩余就是 train

    return tr, va, te


# nnUNet fold0 固定验证集(liver_002,005,...,121,去前导零)
# 来源:/home/pumengyu/nnUNet_result/fold_0/validation_raw_postprocessed/summary.json
_NNUNET_FOLD0_VAL = {
    "liver_2","liver_5","liver_9","liver_12","liver_18","liver_28",
    "liver_44","liver_49","liver_57","liver_58","liver_60","liver_64",
    "liver_69","liver_81","liver_94","liver_98","liver_101","liver_117","liver_121",
}

#全部大写表示常量(constant),常量一般不修改

# test集固定:原split_three_ways(seed=0)的结果
_FIXED_TEST = {
    "liver_107","liver_15","liver_27","liver_36","liver_37","liver_4",
    "liver_40","liver_7","liver_71","liver_77","liver_78","liver_87","liver_92",
}


def split_fixed(pt_paths: list):
    """
    三路划分,
    固定划分,val对齐nnUNet fold0验证集,保证指标可直接比较。

    val: 19个,与nnUNet fold0完全一致
    test: 13个,原split_three_ways(seed=0)的test集
    train: 剩余99个

    用法:tr, va, te = split_fixed(all_pt)
    """
    import os

    va = [p for p in pt_paths if os.path.basename(p).replace(".pt", "") in _NNUNET_FOLD0_VAL]
    te = [p for p in pt_paths if os.path.basename(p).replace(".pt", "") in _FIXED_TEST]
    tr = [p for p in pt_paths if
          os.path.basename(p).replace(".pt", "") not in _NNUNET_FOLD0_VAL and
          os.path.basename(p).replace(".pt", "") not in _FIXED_TEST]
    return tr, va, te


def split_two(pt_paths: list):
    """
    两路划分:train=112个,test=19个,无 val。

    test:  与 nnUNet fold0 验证集完全一致的 19 个,可直接和 nnUNet 指标横向对比
    train: 剩余 112 个全部参与训练,用 train loss 选 best ckpt

    用法:tr, te = split_two(all_pt)
    """
    import os

    te = [p for p in pt_paths if os.path.basename(p).replace(".pt", "") in _NNUNET_FOLD0_VAL]
    tr = [p for p in pt_paths if os.path.basename(p).replace(".pt", "") not in _NNUNET_FOLD0_VAL]
    return tr, te


# 监控子集：从112个训练案例中手工挑选,覆盖无肿瘤/极小/小/中等/大各类别,
# 用于训练过程中定期inference监控dice,选best ckpt。
# 这些案例同时也参与训练,不从训练集中排除。
_MONITOR_SET = {
    # 无肿瘤 (1个)
    "liver_38",
    # 极小 <5k (2个, 选中间值附近)
    "liver_24",   # tumor=1,458
    "liver_62",   # tumor=3,940
    # 小 5k-50k (3个, 覆盖小/中/大端)
    "liver_45",   # tumor=7,028
    "liver_19",   # tumor=12,306
    "liver_26",   # tumor=39,005
    # 中等 50k-300k (2个)
    "liver_82",   # tumor=65,334
    "liver_88",   # tumor=170,931
    # 大 >=300k (2个)
    "liver_56",   # tumor=506,998
    "liver_33",   # tumor=639,212
    # 来自原test集补充 (2个,原test并入train后也需要覆盖)
    "liver_36",   # tumor=58,378 (中等)
    "liver_77",   # tumor=19,405 (小)
}


def split_two_with_monitor(pt_paths: list):
    """
    112个全部参与训练,同时从中抽取有代表性的监控子集用于选best ckpt。

    test:    19个,nnUNet fold0验证集,用于最终和nnUNet指标对比
    train:   112个,全部参与训练(包含monitor子集)
    monitor: 12个,train的子集,覆盖无肿瘤/极小/小/中等/大,
             每隔val_every个epoch在此子集上inference,用dice选best ckpt

    用法: tr, monitor, te = split_two_with_monitor(all_pt)
    """
    import os

    te = [p for p in pt_paths if os.path.basename(p).replace(".pt", "") in _NNUNET_FOLD0_VAL]
    tr = [p for p in pt_paths if os.path.basename(p).replace(".pt", "") not in _NNUNET_FOLD0_VAL]
    monitor = [p for p in tr if os.path.basename(p).replace(".pt", "") in _MONITOR_SET]
    return tr, monitor, te


# /home/PuMengYu/MSD_LiverTumorSeg/medseg_project/medseg/utils/train_utils.py
