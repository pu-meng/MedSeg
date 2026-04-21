import torch
from medseg.data.msd import load_msd_dataset, fixed_split
from medseg.data.transforms import build_train_transforms, build_val_transforms
from monai.data.dataset import Dataset, DataLoader

DATA_ROOT = "/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver"  # 改
PATCH = (96, 96, 96)
VAL_RATIO = 0.2
SEED = 0
N_PRINT = 20  # 打印多少个样本看看


def inspect(loader, name: str):
    print(f"\n=== {name} transformers 检查 ===")
    cnt = 0
    has2 = 0
    for batch in loader:
        y = batch["label"]  # torch tensor after transforms
        if y.ndim == 4:
            y = y.unsqueeze(1)
        y = y.long()
        yy = y[:, 0]

        u = torch.unique(yy).cpu().tolist()
        tumor_vox = int((yy == 2).sum().item())
        liver_vox = int((yy == 1).sum().item())
        flag = tumor_vox > 0
        has2 += int(flag)
        print(
            f"[{cnt:03d}] unique={u} liver_vox={liver_vox} tumor_vox={tumor_vox} has_tumor={flag}"
        )

        cnt += 1
        if cnt >= N_PRINT:
            break

    print(f"printed {cnt} samples, has_tumor_in_printed = {has2}/{cnt}")


def main():
    items, _ = load_msd_dataset(DATA_ROOT)
    tr, va = fixed_split(items, val_ratio=VAL_RATIO, seed=SEED)

    # 关键:不用 CacheDataset,避免缓存干扰,纯检查
    train_ds = Dataset(tr, transform=build_train_transforms(PATCH))
    val_ds = Dataset(va, transform=build_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    inspect(train_loader, "TRAIN")
    inspect(val_loader, "VAL")


if __name__ == "__main__":
    main()
# python -m scripts.check_raw_tumor_labels
