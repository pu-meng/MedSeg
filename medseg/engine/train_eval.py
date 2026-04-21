import torch
import torch.nn as nn
from monai.losses.dice import DiceCELoss
from monai.losses.dice import DiceFocalLoss
from monai.inferers.utils import sliding_window_inference
from monai.losses.tversky import TverskyLoss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss,专为小目标设计。
    L = (1 - TI)^gamma
    TI = TP / (TP + alpha*FP + beta*FN)
    alpha=0.3, beta=0.7: 加大FN惩罚,减少漏检
    gamma=0.75: focal加权,让hard sample贡献更大

    用softmax+one-hot避免sigmoid的数值不稳定(之前tversky出NaN的原因)。
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
        # targets:[B,D,H,W],第三个1 让它变成[B,1,D,H,W]
        # 这里的值是标签0,1,2等,这个的targets_onehot维度是[B,C,D,H,W]
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets.long(), 1)
        # scatter_是散步/分散得意思,将整数标签转换为one-hot编码
        #
        # probs:[B,C,D,H,W]这里的C是总得类别数,B[:,0,...]就是背景类,
        p = probs[:, 1:]  # [:,1:]表示跳过背景类

        t = targets_onehot[:, 1:]
        # p:[B,C-1,D,H,W]这个得p.ndim=5,这里的range(2,5)=[2,3,4]
        dims = tuple(range(2, p.ndim))
        # p:[B,C-1,D,H,W],t:[B,C-1,D,H,W],
        # p,t都是float32,p是预测概率,t是标签的one-hot,值为0或1
        tp = (p * t).sum(dim=dims)  # tp:[B,C-1]
        fp = (p * (1 - t)).sum(dim=dims)
        fn = ((1 - p) * t).sum(dim=dims)
        tversky_index = tp / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = ((1 - tversky_index) ** self.gamma).mean()
        return loss


class DiceCEFocalTverskyLoss(nn.Module):
    """DiceCE + FocalTversky 组合损失，等权求和。
    DiceCE 保证整体分割精度，FocalTversky 强化对漏检(FN)的惩罚提升 Recall。
    weight=0.5 时两者等权，调大 weight 偏向 DiceCE（更高 Precision），调小偏向 FocalTversky（更高 Recall）。
    """
    def __init__(self, ft_alpha=0.3, ft_beta=0.7, ft_gamma=0.75, weight=0.5):
        super().__init__()
        self.weight = weight
        self.dicece = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.focaltversky = FocalTverskyLoss(alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma)

    def forward(self, logits, y):
        return self.weight * self.dicece(logits, y) + (1 - self.weight) * self.focaltversky(logits, y)


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
    """打印 batch 的类型结构,仅用于第一个 batch 的调试"""
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
        return FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=0.75)
    elif loss_type == "dicece_focaltversky":
        return DiceCEFocalTverskyLoss(ft_alpha=0.3, ft_beta=0.7, ft_gamma=0.75, weight=0.5)
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
                print(
                    f"[warn] step={step} NaN/Inf loss, skipping (total skipped={n_nan})"
                )
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
                print(
                    f"[warn] step={step} NaN/Inf loss, skipping (total skipped={n_nan})"
                )
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
        print(
            f"[warn] epoch had {n_nan} NaN/Inf batches skipped out of {n_valid + n_nan} total"
        )
    return running / max(1, n_valid)


def _deep_supervision_loss(loss_fn, logits, y):
    """
    处理 DynUNet deep_supervision=True 时训练模式的多尺度输出。
    logits: [B, N, C, D, H, W](N 个尺度)或普通 [B, C, D, H, W]
    权重方案(nnUNet): w_i = 1/2^i,归一化后加权求和。
    scale 0 是最终输出(权重最大),scale 1+ 是辅助输出。
    DynUNet开启deep_supervision=True,模型训练阶段会输出多个尺度的预测,而不只是最终输出;这个函数负责把这些多尺度的输出的损失加权合并成一个总损失.

    """
    if logits.ndim == 5:
        # 没开深监督,普通前向
        return loss_fn(logits, y)

    n = logits.shape[1]
    weights = [1.0 / (2**i) for i in range(n)]
    w_sum = sum(weights)
    total = sum((w / w_sum) * loss_fn(logits[:, i], y) for i, w in enumerate(weights))
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
    loss_fn=None,  # 外部传入可复用,避免每 epoch 重建
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
            # 有时候batch是list[tuple1],甚至list[list[tuple1]]
            batch = batch[0]
        # 后面用到batch["image"]所以batch必须是字典,
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        # y是真实标签,x是输入图像
        if y.ndim == 4:
            y = y.unsqueeze(1)
        y = y.long()

        # 二分类强约束: 标签只能是 0/1(仅前3个step检查,避免每step cuda sync拖慢训练)
        if step <= 3:
            # 这里的y:[B,1,D,H,W]
            yy = y[:, 0]  # yy:[B,D,H,W],去掉那个1维,因为它没什么用,标签就是0/1的值
            u = torch.unique(yy).detach().cpu().tolist()
            
            tumor_vox = int((yy == 1).sum().item())
            bg_vox = int((yy == 0).sum().item())
            print(
                f"[二分类调试] 第1个batch {step}: 出现的类别={u} bg_vox={bg_vox} tumor_vox={tumor_vox}"
            )
            if not all(v in (0, 1) for v in u):
                raise ValueError(
                    f"Binary tumor training expects labels in {{0,1}}, but got {u}"
                )

        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            logits = model(x)
            loss = _deep_supervision_loss(loss_fn, logits, y)
            if torch.isnan(loss) or torch.isinf(loss): #type:ignore
                n_nan += 1
                print(
                    f"[warn] step={step} NaN/Inf loss, skipping (total skipped={n_nan})"
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()#type:ignore
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(x)
                loss = _deep_supervision_loss(loss_fn, logits, y)
            if torch.isnan(loss) or torch.isinf(loss):  #type:ignore
                n_nan += 1
                print(
                    f"[warn] step={step} NaN/Inf loss, skipping (total skipped={n_nan})"
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        running += float(loss.item())#type:ignore
        n_valid += 1

        if step % 10 == 0:
            print(
                f"[train-binary] step={step}/{len(loader)} loss={float(loss.item()):.4f}"  #type:ignore
            )

    if n_nan > 0:
        print(
            f"[warn] epoch had {n_nan} NaN/Inf batches skipped out of {n_valid + n_nan} total"
        )
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
    n_classes = len(class_ids)
    sum_dice = torch.zeros(n_classes, device="cpu", dtype=torch.float64)
    n_valid_per_class = torch.zeros(n_classes, device="cpu", dtype=torch.int64)
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

            for i, c in enumerate(class_ids):
                p = pred == c  # 预测中属于类别 c 的体素
                g = y == c  # 标签中属于类别 c 的体素

                inter = (p & g).sum(dim=(1, 2, 3)).double()
                denom = p.sum(dim=(1, 2, 3)).double() + g.sum(dim=(1, 2, 3)).double()
                gt_sum = g.sum(dim=(1, 2, 3)).double()

                # 只对 gt 非空的 case 计算 dice；无肿瘤 case 不计入，避免虚高
                valid = gt_sum > 0
                if valid.any():
                    d_valid = 2.0 * inter[valid] / denom[valid].clamp(min=1e-8)
                    sum_dice[i] += d_valid.mean().detach().cpu()
                    n_valid_per_class[i] += 1

            n_batches += 1

            del x, y, logits, pred

    per_class = [
        float(sum_dice[i] / n_valid_per_class[i]) if n_valid_per_class[i] > 0 else float("nan")
        for i in range(n_classes)
    ]
    valid_scores = [s for s in per_class if not (s != s)]  # filter nan
    mean_fg = float(sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0

    print(f"Dice 每个类别:{[round(s, 4) for s in per_class]} (仅统计有gt的case, n={n_valid_per_class.tolist()})")

    return {
        "mean_fg": mean_fg,
        "per_class": [float(x) for x in per_class] if return_per_class else [],
        "class_ids": class_ids,
    }
