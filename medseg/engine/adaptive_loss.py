# ─────────────────────────────────────────────────────────────────────────────
# 可学习权重的二分类损失函数 + 对应的单 epoch 训练函数
#
# 解决的问题：
#   Stage 2 的 label 是二分类（0=liver区域, 1=tumor）。
#   普通 DiceCELoss 只用 include_background=False，仅监督 tumor（前景），
#   模型可能偷懒把所有体素预测为 0（全 liver），loss 仍然不大。
#
#   LearnableWeightedLoss 同时计算 tumor loss 和 liver loss，
#   用一个可学习标量 alpha 自动平衡两者权重，让模型同时学好 liver 和 tumor。
# ─────────────────────────────────────────────────────────────────────────────
import torch

from .train_eval import build_loss_fn_binary


class LearnableWeightedLoss(torch.nn.Module):
    """
    可学习权重的 liver/tumor 双监督损失。

    核心公式：
        total_loss = alpha * loss_tumor + (1 - alpha) * loss_liver

        alpha 是可训练标量，由 sigmoid(log_alpha) 映射到 (0,1)，
        训练过程中自动学习 tumor 和 liver 各自应占的权重。

    为什么用 log_alpha 而不直接学 alpha：
        直接约束 alpha ∈ (0,1) 需要 clamp，梯度不干净。
        用 log_alpha ∈ (-∞, +∞) + sigmoid 映射，无约束优化即可，
        梯度更稳定。初始 log_alpha=0 → sigmoid(0)=0.5，即两者初始等权。
    """

    def __init__(self, base_loss_type: str = "dicece", init_alpha: float = 0.0):
        """
        Args:
            base_loss_type : 基础损失类型，支持 "dicece" / "dicefocal" / "tversky"
            init_alpha     : log_alpha 的初始值，默认 0.0 → alpha=0.5（初始等权）
        """
        super().__init__()

        # log_alpha：可学习标量，注册为 nn.Parameter，会被 optimizer 自动更新
        self.log_alpha = torch.nn.Parameter(torch.tensor(float(init_alpha)))

        # base_loss：具体的损失函数实例（DiceCELoss / DiceFocalLoss / TverskyLoss）
        # 内部设置 include_background=False + sigmoid=True，对二分类前景计算损失
        self.base_loss = build_loss_fn_binary(base_loss_type)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算 liver/tumor 双监督的加权总损失。

        Args:
            logits : 模型原始输出，shape=[B, 2, D, H, W]，未经激活
            y      : 标签，shape=[B, 1, D, H, W]，值域 {0=liver区域, 1=tumor}

        Returns:
            total  : 标量损失值

        为什么需要 y_liver = 1 - y：
            base_loss 设置了 include_background=False，只计算"前景"的损失。
            ① loss_tumor：y 中 tumor=1 是前景，直接传入即可。
            ② loss_liver：需要把 label 翻转（y_liver = 1 - y），
               让 liver=1 成为前景；否则 liver(0) 会被当作背景跳过，
               liver loss 算不到任何东西。
        """
        # alpha ∈ (0,1)：tumor 的损失权重，sigmoid 保证值域合法
        alpha = torch.sigmoid(self.log_alpha)

        # tumor 损失：y 中 1=tumor 是前景
        loss_tumor = self.base_loss(logits, y)

        # liver 损失：翻转 label，让 liver(原来的0) 变成前景(1)
        y_liver = 1 - y
        loss_liver = self.base_loss(logits, y_liver)

        # 加权求和：alpha 自动学习，网络自己决定更关注 tumor 还是 liver
        total = alpha * loss_tumor + (1.0 - alpha) * loss_liver
        return total

    def get_weights(self) -> dict:
        """
        返回当前的 liver/tumor 权重，供训练日志记录。

        Returns:
            {"w_liver": float, "w_tumor": float}，两者之和为 1
        """
        alpha = torch.sigmoid(self.log_alpha).item()
        return {"w_liver": round(1 - alpha, 4), "w_tumor": round(alpha, 4)}


def train_one_epoch_binary_learnable(
    model,
    loader,
    optimizer,           # 模型参数的优化器（AdamW）
    criterion,           # LearnableWeightedLoss 实例
    criterion_optimizer, # criterion 自身参数（log_alpha）的优化器（Adam）
    device,
    scaler=None,         # AMP GradScaler，传 None 则不使用混合精度
    epoch=None,
    epochs=None,
):
    """
    使用 LearnableWeightedLoss 的单 epoch 训练函数。

    与普通训练函数的区别：
        多了一个 criterion_optimizer，每个 step 同时更新两组参数：
        ① optimizer          → 更新模型所有参数（DynUNet 权重）
        ② criterion_optimizer → 更新 criterion.log_alpha（损失权重）
        两者在同一个 forward/backward 中联合优化，log_alpha 和模型参数一起收敛。

    Args:
        model                : 待训练的分割模型
        loader               : DataLoader，每个 batch 是 {"image":..., "label":...}
        optimizer            : 模型参数优化器
        criterion            : LearnableWeightedLoss 实例
        criterion_optimizer  : criterion.log_alpha 的优化器
        device               : "cuda" 或 "cpu"
        scaler               : AMP GradScaler，None 表示不用混合精度
        epoch / epochs       : 当前/总 epoch 数，仅用于打印进度

    Returns:
        float: 本 epoch 所有 step 的平均 loss
    """
    model.train()
    criterion.train()  # 将 criterion 也置为训练模式（影响 dropout 等子模块）

    running = 0.0  # 累加每个 step 的 loss，最后除以 step 数取平均
    n = len(loader)
    print(f"Epoch {epoch}/{epochs} training (binary learnable weight):")

    for step, batch in enumerate(loader, start=1):
        # MONAI 某些 transform 返回 list，拆包取第一个 dict
        while isinstance(batch, list):
            batch = batch[0]

        x = batch["image"].to(device)  # [B, 1, D, H, W] float32
        y = batch["label"].to(device)  # [B, 1, D, H, W] 或 [B, D, H, W]

        # 保证 y 是 [B, 1, D, H, W]：DiceCELoss 要求 label 有 channel 维
        if y.ndim == 4:
            y = y.unsqueeze(1)
        y = y.long()  # DiceCELoss / CrossEntropyLoss 要求 label 为 int64

        # 每个 step 开始前清零梯度（set_to_none=True 比置零更节省显存）
        optimizer.zero_grad(set_to_none=True)
        criterion_optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            # ── 不使用 AMP，全精度 float32 训练 ────────────────────────────
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            criterion_optimizer.step()
        else:
            # ── 使用 AMP（混合精度） ─────────────────────────────────────────
            # autocast：forward 在 float16 下运行，节省显存并加速矩阵运算
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(x)
                loss = criterion(logits, y)
            # scaler.scale：loss 乘以缩放因子，防止 float16 梯度下溢为 0
            scaler.scale(loss).backward()
            scaler.step(optimizer)           # 反缩放梯度 + 更新模型参数
            scaler.step(criterion_optimizer) # 反缩放梯度 + 更新 log_alpha
            scaler.update()                  # 动态调整缩放因子（成功则不变，溢出则缩小）

        running += float(loss.item())

    # 返回本 epoch 所有 step 的平均 loss
    return running / max(1, n)
