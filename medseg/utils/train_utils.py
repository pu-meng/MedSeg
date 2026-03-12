import random
import numpy as np
import torch
import os
from medseg.data.msd import load_msd_dataset
from medseg.data.dataset_offline import load_pt_paths
from medseg.data.build_loader import build_loaders, build_loaders_offline


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_stage_ratios(
    epoch: int, epochs: int, early_ratios=(0.0, 1.0, 0.0), late_ratios=(0.0, 0.5, 0.5)
):
    """
    early_ratios: 前半段使用的采样比例
    late_ratios: 后半段使用的采样比例
    如果 early_ratios == late_ratios, 全程用同一个比例
    """
    cut = int(epochs / 2)
    if epoch <= cut:
        return early_ratios
    else:
        return late_ratios


def build_metrics(
    best, best_epoch, best_c1, best_c2, total_sec, epochs, n_train, n_val
):
    """训练结束后整理最终指标,返回 dict"""
    return {
        "best_score": round(float(best), 4),
        "best_epoch": int(best_epoch),
        "best_liver_dice": None if best_c1 is None else round(float(best_c1), 4),
        "best_tumor_dice": None if best_c2 is None else round(float(best_c2), 4),
        "total_train_hours": round(total_sec / 3600.0, 2),
        "epochs": int(epochs),
        "n_train_cases": int(n_train),
        "n_val_cases": int(n_val),
    }


def build_report(metrics: dict) -> str:
    """把 metrics dict 格式化成人可读的 report 字符串"""
    lines = [
        f"best_epoch:       {metrics['best_epoch']}",
        f"best_liver_dice:  {metrics['best_liver_dice']}",
        f"best_tumor_dice:  {metrics['best_tumor_dice']}",
        f"best_score:       {metrics['best_score']}",
        f"n_train_cases:    {metrics['n_train_cases']}",
        f"n_val_cases:      {metrics['n_val_cases']}",
        f"total_train_hours:{metrics['total_train_hours']}",
    ]
    return "\n".join(lines)


def load_data(args):
    """
    根据是否传入 preprocessed_root 自动选择离线/在线模式
    返回 (tr, va, use_offline)
    离线模式返回路径列表,在线模式返回字典列表
    """
    use_offline = args.preprocessed_root is not None

    if use_offline:
        print(f"[离线模式] 读取 .pt 文件: {args.preprocessed_root}")
        all_pt = load_pt_paths(args.preprocessed_root)

        tr, va, te = split_three_ways(
            all_pt, test_ratio=args.test_ratio, val_ratio=args.val_ratio, seed=args.seed
        )

        if args.train_n and args.train_n > 0:
            tr = tr[: args.train_n]
        if args.val_n and args.val_n > 0:
            va = va[: args.val_n]
    else:
        print(f"[在线模式] 读取 .nii.gz: {args.data_root}")
        train_items, _ = load_msd_dataset(args.data_root)
        tr, va, te =split_three_ways(train_items,val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
        
        if args.train_n and args.train_n > 0:
            tr = tr[: args.train_n]
        if args.val_n and args.val_n > 0:
            va = va[: args.val_n]
        tr_ids = {os.path.basename(x["image"]) for x in tr}
        va_ids = {os.path.basename(x["image"]) for x in va}
        assert len(tr_ids & va_ids) == 0, "train/val overlap!"

    print(f"训练: {len(tr)}  验证: {len(va)},测试: {len(te)}")
    return tr, va, te, use_offline


def build_loaders_auto(args, tr, va, use_offline, ratios):
    """
    根据 use_offline 自动选择离线/在线 loader
    """
    if use_offline:
        return build_loaders_offline(
            tr,
            va,
            patch_size=tuple(args.patch),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratios=ratios,
            prefetch_factor=args.prefetch_factor,
            repeats=args.repeats,
        )
    else:
        return build_loaders(
            tr,
            va,
            patch_size=tuple(args.patch),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_rate=args.cache_rate,
            train_ratios=ratios,
            prefetch_factor=args.prefetch_factor,
        )


def split_three_ways(
    pt_paths: list, test_ratio: float = 0.1, val_ratio: float = 0.2, seed: int = 0
):
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
