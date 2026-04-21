import os
import torch


def save_ckpt(path, model, optimizer, epoch, best_metric):
    """
    决定.pt文件里面存什么内容,
    把训练状态保存成checkpoint文件,也就是.pt
    .pt这里的是一个Python字典

    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_ckpt(path, model, optimizer=None, map_location="cpu"):
    """
    把checkpoint文件恢复到模型
    model.load_state_dict(ckpt["model"], strict=True)
    是加载模型权重;
    optimizer.load_state_dict(ckpt["optim"])
    是恢复优化器状态
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    state_dict = ckpt["model"]
    ckpt_has_prefix = any(k.startswith("_orig_mod.") for k in state_dict)
    model_has_prefix = any(k.startswith("_orig_mod.") for k in model.state_dict())
    if ckpt_has_prefix and not model_has_prefix:
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    elif not ckpt_has_prefix and model_has_prefix:
        state_dict = {"_orig_mod." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    if optimizer is not None and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    return ckpt


def save_ckpt_full(
    path,
    model,
    optimizer,
    epoch,
    best_metric,
    scheduler=None,
    scaler=None,
    best_epoch=None,
):
    """
    two-stage训练使用的完整checkpoint
    不影响原来的save_ckpt
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }

    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    if best_epoch is not None:
        ckpt["best_epoch"] = best_epoch

    torch.save(ckpt, path)



def load_init_weights(path, model, map_location="cpu"):
    """
    从 path 加载 checkpoint，筛选「名字相同且形状相同」的参数复制到 model，
    其他不匹配的层保持原样（随机初始化或已有的值）。
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    src = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(k.startswith("_orig_mod.") for k in src):
        src = {k.removeprefix("_orig_mod."): v for k, v in src.items()}

    dst = model.state_dict()
    matched = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    # dynunet_ca 的 backbone 参数名带 backbone. 前缀，裸 ckpt 里没有，尝试加前缀重映射
    if len(matched) == 0:
        remapped = {"backbone." + k: v for k, v in src.items()}
        matched = {
            k: v for k, v in remapped.items() if k in dst and dst[k].shape == v.shape
        }
    dst.update(matched)
    model.load_state_dict(dst, strict=False)
    print(f"[init] loaded {len(matched)} matched params from {path}")


def load_ckpt_full(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
):
    """
    two-stage训练使用的完整resume
    """

    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    state_dict = ckpt["model"]
    ckpt_has_prefix = any(k.startswith("_orig_mod.") for k in state_dict)
    model_has_prefix = any(k.startswith("_orig_mod.") for k in model.state_dict())
    if ckpt_has_prefix and not model_has_prefix:
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    elif not ckpt_has_prefix and model_has_prefix:
        state_dict = {"_orig_mod." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    if optimizer is not None and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])

    if scheduler is not None and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except (KeyError, ValueError):
            # scheduler state_dict 与当前版本不兼容时跳过,
            # 手动 step 追到已训练的 epoch, 使 LR 与中断时一致
            resumed_epoch = int(ckpt.get("epoch", 0))
            for _ in range(resumed_epoch):
                scheduler.step()
            print(f"[resume] scheduler state incompatible, fast-forwarded {resumed_epoch} steps")

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt