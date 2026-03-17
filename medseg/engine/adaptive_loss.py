# ────────────────────────────────────────────
# 新增：可学习权重的二分类损失 + 对应训练函数
# ────────────────────────────────────────────
import torch

from .train_eval import build_loss_fn_binary
class LearnableWeightedLoss(torch.nn.Module):
    """
   这里的输入init_alpha是1个值,
   torch.nn.Parameter()自动注册为模型参数,optimizer会更新它
   sigmoid(x)=1/(1+e^(-x))

    """
    def __init__(self, base_loss_type="dicece", init_alpha=0.0):
       
        super().__init__()
      
        self.log_alpha = torch.nn.Parameter(torch.tensor(float(init_alpha)))
        self.base_loss = build_loss_fn_binary(base_loss_type)

    def forward(self, logits, y):
        """
        logits: [B, 2, D,H, W]
        我们这个是二分类问题,所以1维是2,logits是模型的原始输出
        y:标签
        shape=[B,1,D,H,W]
        每个体素的值是0或1
        0=liver,1=tumor
        loss_tumor是一个标量,DiceCELoss内部会把y转成one-hot,和logits比较
        计算结果是这个batch的平均损失值,比如tensor(0.7832)

        为什么需要y_liver=1-y
        因为DiceCELoss设置了include_background=False,只算前景即tumor
        如果不把y翻转, DiceCELoss会认为0是前景,1是背景,所以会算错

        """
        alpha = torch.sigmoid(self.log_alpha)   # tumor权重
        
        # 整体loss（include_background=False，只算前景即tumor）
        loss_tumor = self.base_loss(logits, y)
        
        # 构造liver专用loss（把label翻转：0→1, 1→0）
        y_liver = 1 - y  # liver当前景
        loss_liver = self.base_loss(logits, y_liver)
        
        total = alpha * loss_tumor + (1.0 - alpha) * loss_liver
        return total

    def get_weights(self):
        alpha = torch.sigmoid(self.log_alpha).item()
        return {"w_liver": round(1 - alpha, 4), "w_tumor": round(alpha, 4)}


def train_one_epoch_binary_learnable(
    model,
    loader,
    optimizer,
    criterion,           # LearnableWeightedLoss 实例
    criterion_optimizer, # criterion 自己的优化器
    device,
    scaler=None,
    epoch=None,
    epochs=None,
):
    """
   criterion是LearnableWeightedLoss的实例
   running是累加器,把每个step的loss加起来,最后除以总step数
   正常情况下,batch是字典
   batch={
   "image":tensor[B,C,D,H,W],
   "label":tensor[B,1,D,H,W]
   }
   但是某些情况下,batch是列表,列表里只有一个字典

   y=y.long()把y的数据类型转为'int64'

   CrossEntropyLoss/DiceCELoss要求label必须是int64

   .zero_grad(set_to_none=True)
   在每次反向传播前把上一步的梯度清零,set_to_none=True比默认的更节省内存

   criterion_optimizer=torch.optim.AdamW(
   params=criterion.parameters(),lr=1e-3
   )
   optimizer更新model的所有参数
   criterion_optimizer只更新criterion的参数

    """
    model.train()
    criterion.train()

    running = 0.0
    n = len(loader)
    print(f"Epoch {epoch}/{epochs} training (binary learnable weight):")

    for step, batch in enumerate(loader, start=1):
        while isinstance(batch, list):
            batch = batch[0]

        x = batch["image"].to(device)
        y = batch["label"].to(device)
        if y.ndim == 4:
            y = y.unsqueeze(1)
        y = y.long()

        optimizer.zero_grad(set_to_none=True)
        criterion_optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            criterion_optimizer.step()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(criterion_optimizer)
            scaler.update()

        running += float(loss.item())

      

    return running / max(1, n)