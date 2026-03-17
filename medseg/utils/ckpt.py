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
    model.load_state_dict(ckpt["model"], strict=True)
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

    model.load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt