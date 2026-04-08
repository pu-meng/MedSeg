import torch
import torch.nn as nn
from monai.losses.dice import DiceCELoss
from monai.losses.dice import DiceFocalLoss
from monai.inferers.utils import sliding_window_inference
from monai.losses.tversky import TverskyLoss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss，专为小目标设计。
    L = (1 - TI)^gamma
    TI = TP / (TP + alpha*FP + beta*FN)
    alpha=0.3, beta=0.7: 加大FN惩罚，减少漏检
    gamma=0.75: focal加权，让hard sample贡献更大

    用softmax+one-hot避免sigmoid的数值不稳定（之前tversky出NaN的原因）。
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        # logits: [B, C, D, H, W], targets: [B, 1, D, H, W]
        # 强制 float32 防止 AMP fp16 下 softmax/除法 溢出导致 NaN
        logits = logits.float()
        probs = torch.softmax(logits, dim=1)  # softmax避免sigmoid NaN
        # one-hot: [B, C, D, H, W]
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets.long(), 1)

        # 向量化所有前景类，避免 Python for 循环的 CUDA sync 开销
        # p/t: [B, n_fg, D, H, W]，其中 n_fg = n_classes - 1
        p = probs[:, 1:]
        t = targets_onehot[:, 1:]
        dims = tuple(range(2, p.ndim))  # 对空间维度求和，保留 B 和 C 维度
        tp = (p * t).sum(dim=dims)
        fp = (p * (1 - t)).sum(dim=dims)
        fn = ((1 - p) * t).sum(dim=dims)
        tversky_index = tp / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = ((1 - tversky_index) ** self.gamma).mean()
        return loss

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
    elif loss_type == "focaltversky":
        return FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def build_loss_fn_binary(loss_type="dicece"):
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
            lambda_focal=1.0,
        )
    elif loss_type == "tversky":
        return TverskyLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            alpha=alpha,
            beta=beta,
        )
    elif loss_type == "focaltversky":
        return FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_one_epoch_softmax(
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
    n_valid = 0
    n_nan = 0

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
            if torch.isnan(loss) or torch.isinf(loss):
                n_nan += 1
                print(f"[warn] step={step} NaN/Inf loss, skipping (total skipped={n_nan})")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            if torch.isnan(loss) or torch.isinf(loss):
                n_nan += 1
                print(f"[warn] step={step} NaN/Inf loss, skipping (total skipped={n_nan})")
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        running += float(loss.item())
        n_valid += 1

        if step % 10 == 0:
            print(
                f"[train-multi] step={step}/{len(loader)} loss={float(loss.item()):.4f}"
            )

    if n_nan > 0:
        print(f"[warn] epoch had {n_nan} NaN/Inf batches skipped out of {n_valid + n_nan} total")
    return running / max(1, n_valid)


def _deep_supervision_loss(loss_fn, logits, y):
    """
    处理 DynUNet deep_supervision=True 时训练模式的多尺度输出。
    logits: [B, N, C, D, H, W]（N 个尺度）或普通 [B, C, D, H, W]
    权重方案（nnUNet）: w_i = 1/2^i，归一化后加权求和。
    scale 0 是最终输出（权重最大），scale 1+ 是辅助输出。
    """
    if logits.ndim == 5:
        # 没开深监督，普通前向
        return loss_fn(logits, y)
    # logits: [B, N, C, D, H, W]
    n = logits.shape[1]
    weights = [1.0 / (2 ** i) for i in range(n)]
    w_sum = sum(weights)
    total = sum(
        (w / w_sum) * loss_fn(logits[:, i], y)
        for i, w in enumerate(weights)
    )
    return total


def train_one_epoch_sigmoid_binary(
    model,
    loader,
    optimizer,
    device,
    scaler=None,
    loss_type="dicece",
    epoch=None,
    epochs=None,
    loss_fn=None,  # 外部传入可复用，避免每 epoch 重建
):
    model.train()
    if loss_fn is None:
        loss_fn = build_loss_fn_binary(loss_type)

    running = 0.0
    n_valid = 0
    n_nan = 0

    print(f"Epoch {epoch}/{epochs} training (binary tumor):")

    for step, batch in enumerate(loader, start=1):
        if step == 1:
            _debug_batch_type(batch)

        while isinstance(batch, list):
            batch = batch[0]

        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        if y.ndim == 4:
            y = y.unsqueeze(1)
        y = y.long()

        # 二分类强约束: 标签只能是 0/1（仅前3个step检查，避免每step cuda sync拖慢训练）
        if step <= 3:
            yy = y[:, 0]
            u = torch.unique(yy).detach().cpu().tolist()
            tumor_vox = int((yy == 1).sum().item())
            bg_vox = int((yy == 0).sum().item())
            print(
                f"[debug-binary] batch {step}: unique={u} bg_vox={bg_vox} tumor_vox={tumor_vox}"
            )
            if not all(v in (0, 1) for v in u):
                raise ValueError(
                    f"Binary tumor training expects labels in {{0,1}}, but got {u}"
                )

        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            logits = model(x)
            loss = _deep_supervision_loss(loss_fn, logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                n_nan += 1
                print(f"[warn] step={step} NaN/Inf loss, skipping (total skipped={n_nan})")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(x)
                loss = _deep_supervision_loss(loss_fn, logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                n_nan += 1
                print(f"[warn] step={step} NaN/Inf loss, skipping (total skipped={n_nan})")
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        running += float(loss.item())
        n_valid += 1

        if step % 10 == 0:
            print(
                f"[train-binary] step={step}/{len(loader)} loss={float(loss.item()):.4f}"
            )

    if n_nan > 0:
        print(f"[warn] epoch had {n_nan} NaN/Inf batches skipped out of {n_valid + n_nan} total")
    return running / max(1, n_valid)


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

                # denom==0 表示预测和标签都为空（如无肿瘤case且预测也无肿瘤），视为完美预测dice=1.0
                d = torch.where(denom == 0, torch.ones_like(inter), 2.0 * inter / denom)

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
