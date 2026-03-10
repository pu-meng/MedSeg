import argparse
import torch
from medseg.data.msd import load_msd_dataset, fixed_split
from medseg.data.build_loader import build_loaders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--patch", type=int, nargs=3, default=[96, 96, 96])
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    train_items, _ = load_msd_dataset(args.data_root)
    tr, va = fixed_split(train_items, val_ratio=args.val_ratio)

    train_loader, val_loader = build_loaders(
        tr, va,
        patch_size=tuple(args.patch),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_rate=0.2,
    )

    batch = next(iter(train_loader))
    x = batch["image"]
    y = batch["label"]
    print("train batch image:", x.shape)
    print("train batch label:", y.shape)
    vbatch = next(iter(val_loader))
    vx, vy = vbatch["image"], vbatch["label"]
    print("val sample image:", tuple(vx.shape), vx.dtype)
    print("val sample label:", tuple(vy.shape), vy.dtype)

    print("cuda available:", torch.cuda.is_available())

if __name__ == "__main__":
    main()