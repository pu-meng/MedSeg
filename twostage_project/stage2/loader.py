import os
import glob
from monai.data import DataLoader, CacheDataset
from stage2.transforms import build_stage2_train_transforms, build_stage2_val_transforms


def load_stage2_items(crops_dir):
    images = sorted(glob.glob(os.path.join(crops_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(crops_dir, "labelsTr", "*.nii.gz")))

    assert len(images) == len(labels) and len(images) > 0, \
        f"crops_dir 数据有问题: images={len(images)}, labels={len(labels)}"

    items = []
    for img, lab in zip(images, labels):
        sid = os.path.basename(img).replace(".nii.gz", "")
        items.append({"image": img, "label": lab, "id": sid})
    return items


def build_loaders(tr_items, va_items, patch_size, batch_size,
                  num_workers, prefetch_factor, ratios):
    train_tf = build_stage2_train_transforms(patch_size, ratios=ratios)
    val_tf   = build_stage2_val_transforms()

    train_ds = CacheDataset(tr_items, transform=train_tf,
                            cache_rate=0.0, num_workers=num_workers)
    val_ds   = CacheDataset(va_items, transform=val_tf,
                            cache_rate=0.5, num_workers=num_workers)

    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=(num_workers > 0),
              prefetch_factor=prefetch_factor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, **kw)
    return train_loader, val_loader