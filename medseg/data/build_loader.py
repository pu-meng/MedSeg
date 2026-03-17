from monai.data.dataloader import DataLoader
from monai.data.dataset import CacheDataset
from medseg.data.transforms import build_train_transforms, build_val_transforms
from medseg.data.dataset_offline import OfflineDataset
from medseg.data.transforms_offline import (
    build_train_transforms as build_train_transforms_offline,
    build_val_transforms as build_val_transforms_offline,
)
from torch.utils.data import DataLoader as TorchDataLoader


def build_loaders(
    train_items,
    val_items,
    patch_size=(96, 96, 96),
    batch_size=2,
    num_workers=4,
    cache_rate=0.2,
    train_ratios=None,  # ✅ 新增:动态采样 ratios
    cache_rate_train=None,  # ✅ 新增:train 单独 cache
    cache_rate_val=None,
    prefetch_factor=4,  # ✅ 新增:val 单独 cache
    force_no_cache_for_train=False,
):
    """
        build_loader.py

        负责把 train/val 数据包装成 DataLoader
        在线模式: build_loaders
        离线模式: build_loaders_offline

        说明:
        train_ratios控制训练集随机裁块的时候,不同类别的采样比例.


        CacheDataset 的输入
        train_items=[
        {
        "image":"/path/image1.nii.gz",
        "label":"/path/label1.nii.gz"
        },
        {
        "image":"/path/image2.nii.gz",
        "label":"/path/label2.nii.gz"
        },
        ...
        ]
        为啥这样?
        因为transform按照key操作,例如:
        LoadImaged(keys=["image", "label"])
        意思是读取dict["image"]和dict["label"]的路径
        CacheDataset第一步:拿到一个item,item=train_items[0]
        第二步:拿到item["image"]和item["label"]的路径
        第三步:LoadImaged(keys=["image", "label"])读取路径,得到image和label的tensor
        第四步:transform(image, label)对image和label进行各种操作,例如裁剪,resize等
        第五步:将image和label打包成一个dict,返回给DataLoader
        第六步:DataLoader将dict打包成一个batch,返回给模型

        pin_memory=True,CPU到GPU的拷贝速度更快
        persistent_workers=True,worker进程不会每个epoch都重启,
        worker一直活着


        train_ids=CacheDataset(train_items,...)
        train_loader=DataLoader(train_ids,...)

        train_items = [
        {"image": "/path/xxx1.nii.gz", "label": "/path/yyy1.nii.gz"},
        {"image": "/path/xxx2.nii.gz", "label": "/path/yyy2.nii.gz"},
    ]
        Dataset类似后厨,DataLoader类似服务员,
        train_ids的类型是monai.data.CacheDataset,
        而CacheDataset又继承与torch.utils.data.Dataset,
        所以train_ids本质上是一个Pytorch Dataset
        Dataset="一个可以按索引取样本的数据集合"；
        Dataset必须实现__getitem__()和__len__()方法
        医学中:
        train_ds=train_items+transform pipeline+cache
        可以理解为
        train_ds={
        "data":train_items,
        "transform":transform pipeline,
        "cache":cache
        }
        真正的数据是在调用train_ds[0]的时候才加载的,例如:
        DataLoader只会做
        batch=[
        dataset[i],
        dataset[i+1],
        ...

        ]
        train_items=数据清单
        dataset=单个样本生成器
        DataLoader=批量样本生成器
        医学图像训练系统：
        数据描述，样本生成,批处理



    """

    # 默认 ratios(你原来用的)
    if train_ratios is None:
        train_ratios = (0.0, 0.6, 0.4)

    # train/val cache 分开控制
    if cache_rate_train is None:
        cache_rate_train = cache_rate
    if cache_rate_val is None:
        cache_rate_val = cache_rate

    if force_no_cache_for_train:
        cache_rate_train = 0.0

    # ✅ 动态 ratios 的时候,强制 train 不缓存(否则 ratios 不生效)

    train_tf = build_train_transforms(patch_size, ratios=train_ratios)

    val_tf = build_val_transforms()

    train_ds = CacheDataset(
        data=train_items,
        transform=train_tf,
        cache_rate=cache_rate_train,
        num_workers=num_workers,
    )

    val_ds = CacheDataset(
        data=val_items,
        transform=val_tf,
        cache_rate=cache_rate_val,
        num_workers=num_workers,
    )
    if num_workers > 0:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    return train_loader, val_loader


def build_loaders_offline(
    tr_paths,
    va_paths,
    patch_size=(144, 144, 144),
    batch_size=2,
    num_workers=4,
    train_ratios=None,
    prefetch_factor=4,
    repeats=1,
    merge_label12_to1=False,
):
    """
    离线预处理版 DataLoader.
    输入是 .pt 文件路径列表(由 preprocess_offline.py 生成).
    原 build_loaders 完全不动,两套并存.
    """
    if train_ratios is None:
        train_ratios = (0.0, 1.0)

    train_tf = build_train_transforms_offline(patch_size, ratios=train_ratios)
    val_tf = build_val_transforms_offline()

    train_loader = None
    if len(tr_paths) > 0:
        train_ds = OfflineDataset(
            tr_paths,
            transform=train_tf,
            repeats=repeats,
            merge_label12_to1=merge_label12_to1,
        )
        if num_workers > 0:
            train_loader = TorchDataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=prefetch_factor,
            )
        else:
            train_loader = TorchDataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

    val_ds = OfflineDataset(
        va_paths, transform=val_tf, repeats=1, merge_label12_to1=merge_label12_to1
    )
    if num_workers > 0:
        val_loader = TorchDataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
    else:
        val_loader = TorchDataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    return train_loader, val_loader
