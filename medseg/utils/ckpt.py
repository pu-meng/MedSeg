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
