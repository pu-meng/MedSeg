import torch
from monai.losses.dice import DiceCELoss
from monai.losses.dice import DiceFocalLoss
from monai.inferers.utils import sliding_window_inference
from monai.losses.tversky import TverskyLoss

"""

前景:有人体结构的区域
背景:没有人体结构的区域,例如空气、床板等
_debug_batch_type(batch),在第一个batch时打印batch的类型结构,帮助调试数据加载和transforms是否正确.
batch=DataLoader的一次返回的数据
batch={
"image": torch.Tensor [B, C, D, H, W],
"label": torch.Tensor [B, 1, D, H, W],
}

在线模式和离线模式都需要这个train_eval.py
train.py
│
├── parse_args()             解析参数
│
├── load_data(args)
│       │
│       ├── 在线数据 (.nii.gz)
│       │
│       └── 离线数据 (.pt)
│
├── build_loaders_auto()
│       │
│       ├── Dataset
│       └── DataLoader
│
├── build_model()
│
├── optimizer + scheduler
│
└── 训练循环
        │
        ├── train_one_epoch()          ← train_eval.py
        │
        ├── validate_sliding_window()  ← train_eval.py
        │
        └── save_ckpt()
因为训练逻辑和数据来源没有关系,
最核心的训练代码只有train_one_epoch()和validate_sliding_window()这两个函数,
换模型:build_model()
换数据:build_loaders_auto()
换验证指标:validate_sliding_window()
换损失函数:train_one_epoch()

"""


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


def build_loss_fn_multiclass(loss_type="dicece"):
    alpha = 0.3
    beta = 0.7

    if loss_type == "dicece":
        return DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        )
    elif loss_type == "dicefocal":
        return DiceFocalLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            gamma=2.0,
            lambda_dice=1.0,
            lambda_focal=2.0,
        )
    elif loss_type == "tversky":
        return TverskyLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            alpha=alpha,
            beta=beta,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def build_loss_fn_binary(loss_type="dicece"):
    alpha = 0.3
    beta = 0.7

    if loss_type == "dicece":
        return DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            sigmoid=True,
        )
    elif loss_type == "dicefocal":
        return DiceFocalLoss(
            include_background=False,
            to_onehot_y=True,
            sigmoid=True,
            gamma=2.0,
            lambda_dice=1.0,
            lambda_focal=2.0,
        )
    elif loss_type == "tversky":
        return TverskyLoss(
            include_background=False,
            to_onehot_y=True,
            sigmoid=True,
            alpha=alpha,
            beta=beta,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_one_epoch_multiclass(
    model,
    loader,
    optimizer,
    device,
    scaler=None,
    loss_type="dicece",
    epoch=None,
    epochs=None,
):
    model.train()
    loss_fn = build_loss_fn_multiclass(loss_type)

    running = 0.0
    n = len(loader)

    print(f"Epoch {epoch}/{epochs} training (multiclass):")

    for step, batch in enumerate(loader, start=1):
        if step == 1:
            _debug_batch_type(batch)

        while isinstance(batch, list):
            batch = batch[0]

        x = batch["image"].to(device)
        y = batch["label"].to(device)

        if y.ndim == 4:
            y = y.unsqueeze(1)
        y = y.long()

        if step <= 3:
            yy = y[:, 0]
            u = torch.unique(yy).detach().cpu().tolist()
            liver_vox = int((yy == 1).sum().item())
            tumor_vox = int((yy == 2).sum().item())
            print(
                f"[debug-multi] batch {step}: unique={u} liver_vox={liver_vox} tumor_vox={tumor_vox}"
            )

        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        running += float(loss.item())

        if step % 10 == 0:
            print(
                f"[train-multi] step={step}/{len(loader)} loss={float(loss.item()):.4f}"
            )

    return running / max(1, n)


def train_one_epoch_binary(
    model,
    loader,
    optimizer,
    device,
    scaler=None,
    loss_type="dicece",
    epoch=None,
    epochs=None,
):
    model.train()
    loss_fn = build_loss_fn_binary(loss_type)

    running = 0.0
    n = len(loader)

    print(f"Epoch {epoch}/{epochs} training (binary tumor):")

    for step, batch in enumerate(loader, start=1):
        if step == 1:
            _debug_batch_type(batch)

        while isinstance(batch, list):
            batch = batch[0]

        x = batch["image"].to(device)
        y = batch["label"].to(device)

        if y.ndim == 4:
            y = y.unsqueeze(1)
        y = y.long()
        u = torch.unique(y)
        if not torch.all((u == 0) | (u == 1)):
            raise ValueError(
                f"Binary tumor training expects labels in {{0,1}}, but got {u.detach().cpu().tolist()}"
            )

        # 二分类强约束: 标签只能是 0/1
        if step <= 3:
            yy = y[:, 0]
            u = torch.unique(yy).detach().cpu().tolist()
            tumor_vox = int((yy == 1).sum().item())
            bg_vox = int((yy == 0).sum().item())
            print(
                f"[debug-binary] batch {step}: unique={u} bg_vox={bg_vox} tumor_vox={tumor_vox}"
            )

        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        running += float(loss.item())

        if step % 10 == 0:
            print(
                f"[train-binary] step={step}/{len(loader)} loss={float(loss.item()):.4f}"
            )

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
    训练时候:
    x=batch["image"].to(device)  # [B, C, D, H, W]
    logits=model(x)
    loss=loss_fn(logits, y)
    这里的x是DataLoader提前裁好的patch,
    没有sliding_window_inference(),也没有roi_size、sw_batch_size、overlap这些参数,
    训练不用滑窗,验证才用滑窗,因为验证时要评测整个体积的dice,而不是patch的dice.
    overlap只在验证里有意义,因为overlap控制的是:滑窗窗口之间重叠多少


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

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = sliding_window_inference(
                    inputs=x,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                )
            logits = logits.float()  # type:ignore

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
