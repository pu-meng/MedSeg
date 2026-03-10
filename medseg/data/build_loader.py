from monai.data import DataLoader, CacheDataset
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
):
    """
    说明:
    - CacheDataset 会缓存 transform 的输出.
      如果你想每个 epoch 动态更换 RandCropByLabelClassesd 的 ratios,
      那么 train 一侧必须 cache_rate=0.0,否则 ratios 改了也不会生效.
    - val transforms 固定,可以继续缓存(cache_rate_val 可保持原值).
    """

    # 默认 ratios(你原来用的)
    if train_ratios is None:
        train_ratios = (0.0, 0.6, 0.4)

    # train/val cache 分开控制
    if cache_rate_train is None:
        cache_rate_train = cache_rate
    if cache_rate_val is None:
        cache_rate_val = cache_rate

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

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0),
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
    repeats=1
):
    """
    离线预处理版 DataLoader.
    输入是 .pt 文件路径列表(由 preprocess_offline.py 生成).
    原 build_loaders 完全不动,两套并存.
    """
    if train_ratios is None:
        train_ratios = (0.0, 0.05, 0.95)

    train_tf = build_train_transforms_offline(patch_size, ratios=train_ratios)
    val_tf = build_val_transforms_offline()

    train_loader = None
    if len(tr_paths) > 0:
        train_ds = OfflineDataset(tr_paths, transform=train_tf, repeats=repeats)
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

    val_ds = OfflineDataset(va_paths, transform=val_tf, repeats=1)
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
