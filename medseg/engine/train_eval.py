import torch


from monai.losses.dice import DiceCELoss

from monai.losses.dice import DiceFocalLoss

from monai.inferers.utils import sliding_window_inference


from monai.losses.tversky import TverskyLoss


def _debug_batch_type(batch):
    """打印 batch 的类型结构，仅用于第一个 batch 的调试"""
    print("type(batch) =", type(batch))
    if isinstance(batch, dict):
        print("batch.keys() =", batch.keys())
        return
    if not isinstance(batch, list):
        return
    print("len(batch) =", len(batch))
    if not batch:
        return
    first = batch[0]
    print("type(batch[0]) =", type(first))
    if isinstance(first, dict):
        print("batch[0].keys() =", first.keys())
    elif isinstance(first, list):
        print("len(batch[0]) =", len(first))
        if first:
            print("type(batch[0][0]) =", type(first[0]))


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler=None,
    loss_type="dicece",
    epoch=None,
    epochs=None,
):
    """
    alpha:weight of the false positives
    beta:weight of the false negatives
    """
    model.train()
    alpha = 0.3
    beta = 0.7

    if loss_type == "dicece":
        loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    elif loss_type == "dicefocal":
        loss_fn = DiceFocalLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            gamma=2.0,
            lambda_dice=1.0,
            lambda_focal=2.0,
        )

    elif loss_type == "tversky":
        loss_fn = TverskyLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            alpha=alpha,
            beta=beta,
        )
    elif loss_type == "focaltversky":
        try:
            loss_fn = TverskyLoss(
                include_background=False,
                to_onehot_y=True,
                softmax=True,
                alpha=alpha,
                beta=beta,
                gamma=0.75,
                # 如果你的 monai 不支持,会走 except
            )
        except TypeError:
            print(
                "[warn] TverskyLoss(gamma=..) not supported in this MONAI version, fallback to plain TverskyLoss"
            )
            loss_fn = TverskyLoss(
                include_background=False,
                to_onehot_y=True,
                softmax=True,
                alpha=alpha,
                beta=beta,
            )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    running = 0.0
    n = len(loader)

    # 使用print直接显示 epoch 和训练进度
    print(f"Epoch {epoch}/{epochs} training:")

    for step, batch in enumerate(loader, start=1):
        if step == 1:
            _debug_batch_type(batch)
        while isinstance(batch, list):
            batch = batch[0]
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        if y.ndim == 4:
            y = y.unsqueeze(1)
        y = y.long().to(device)

        # 可选:只在前5个batch打印label情况(避免刷屏)
        if step <= 5:
            yy = y[:, 0]
            u = torch.unique(yy).detach().cpu().tolist()
            tumor_vox = int((yy == 2).sum().item())
            liver_vox = int((yy == 1).sum().item())
            print(
                f"[debug] batch {step}: unique={u} tumor_vox={tumor_vox} liver_vox={liver_vox}"
            )

        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        running += float(loss.item())

        # 每20个batch打印一次loss,控制输出频率
        if step % 10 == 0:
            print(f"[train] step={step}/{len(loader)} loss={float(loss.item()):.4f}")

    # 计算并返回当前epoch的平均loss
    return running / max(1, n)


def validate_sliding_window(
    model,
    loader,
    device,
    roi_size=(96, 96, 96),
    sw_batch_size=2,
    num_classes=3,
    overlap=0.5,
    return_per_class=True,
):
    """
    logits shape:[1,3,D,H,W],3是类别数

    """

    model.eval()

    eps = 1e-8
    class_ids = list(range(1, num_classes))
    sum_dice = torch.zeros(len(class_ids), device="cpu", dtype=torch.float64)
    n_batches = 0

    with torch.no_grad():
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

            assert isinstance(logits, torch.Tensor)

            pred = torch.argmax(logits, dim=1)

            dices = []

            for c in class_ids:
                p = pred == c  # 预测中属于类别 c 的体素
                g = y == c  # 标签中属于类别 c 的体素

                inter = (p & g).sum(dim=(1, 2, 3)).double()
                denom = p.sum(dim=(1, 2, 3)).double() + g.sum(dim=(1, 2, 3)).double()

                d = (2.0 * inter + eps) / (denom + eps)

                dices.append(d.mean().detach().cpu())

            dices = torch.stack(dices)

            sum_dice += dices
            n_batches += 1

            del x, y, logits, pred

    per_class = (sum_dice / max(1, n_batches)).tolist()
    mean_fg = float(sum(per_class) / max(1, len(per_class)))

    print(f"Dice 每个类别:{[float(x) for x in per_class]}")

    return {
        "mean_fg": mean_fg,
        "per_class": [float(x) for x in per_class] if return_per_class else [],
        "class_ids": class_ids,
    }
