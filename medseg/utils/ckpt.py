import os
import torch

def save_ckpt(path, model, optimizer, epoch, best_metric):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"model": model.state_dict(),
         "optim": optimizer.state_dict(),
         "epoch": epoch,
         "best_metric": best_metric},
        path,
    )

def load_ckpt(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location,weights_only=True)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    return ckpt