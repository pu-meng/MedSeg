import torch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference


def train_one_epoch(model, loader, optimizer, device, scaler, loss_fn, epoch, epochs):
    model.train()
    running, n = 0.0, len(loader)
    print(f"Epoch {epoch}/{epochs} [Stage2 tumor]")

    for it, batch in enumerate(loader, 1):
        while isinstance(batch, list):
            batch = batch[0]

        x = batch["image"].to(device)
        y = batch["label"].to(device)
        if y.ndim == 4:
            y = y.unsqueeze(1)
        y = y.long()

        if it <= 3:
            tumor_vox = int((y == 1).sum().item())
            print(f"  [debug] batch {it}: tumor_vox={tumor_vox}")

        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            logits = model(x)
            loss   = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss   = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        running += float(loss.item())
        if it % 4 == 0:
            print(f"  it={it}/{n}  loss={loss.item():.4f}")

    return running / max(1, n)


@torch.no_grad()
def validate(model, loader, device, roi_size, sw_batch_size, overlap):
    model.eval()
    eps = 1e-8
    sum_dice, n_batches = 0.0, 0

    for batch in loader:
        while isinstance(batch, list):
            batch = batch[0]

        x = batch["image"].to(device)
        y = batch["label"]
        if y.ndim == 4:
            y = y.unsqueeze(1)
        y = y[:, 0].long().to(device)

        logits = sliding_window_inference(
            inputs=x,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
        pred = torch.argmax(logits, dim=1)

        p = pred == 1
        g = y    == 1
        inter = (p & g).sum(dim=(1, 2, 3)).double()
        denom = p.sum(dim=(1, 2, 3)).double() + g.sum(dim=(1, 2, 3)).double()
        dice  = (2.0 * inter + eps) / (denom + eps)

        sum_dice  += dice.mean().item()
        n_batches += 1
        del x, y, logits, pred

    tumor_dice = sum_dice / max(1, n_batches)
    print(f"  [val] tumor_dice={tumor_dice:.4f}")
    return tumor_dice


def build_loss(loss_type="dicece"):
    if loss_type == "dicece":
        return DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    raise ValueError(f"未知 loss: {loss_type}")