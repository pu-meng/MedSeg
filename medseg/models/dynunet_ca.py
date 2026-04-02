import torch
import torch.nn as nn
from medseg.models.dynunet import build_dynunet


class ChannelAttn3D(nn.Module):
    """
    CBAM Channel Attention (3D).CBAM=Convolutional Block Attention Module
    同时用 Global Average Pooling 和 Global Max Pooling,
    共享同一 MLP,相加后 sigmoid 得到通道权重。
    参数量 = 2*(C*(C//r) + (C//r)*C)
    这个的特点是avg_out+max_out
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[0], x.shape[1]
        avg_out = self.fc(x.mean(dim=(2, 3, 4)))        # [B, C]
        max_out = self.fc(x.amax(dim=(2, 3, 4)))        # [B, C]
        scale = torch.sigmoid(avg_out + max_out)        # [B, C]
        return x * scale.view(B, C, 1, 1, 1)


class SpatialAttn3D(nn.Module):
    """
    CBAM Spatial Attention (3D).
    沿通道维度做 AvgPool+MaxPool,concat 后 Conv3d(2->1) 产生空间权重图。
    参数量 = 2 * kernel_size^3(无 bias)
    这个的特点是torch.cat(([avg_out,max_out],dim=1))
    
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)           # [B, 1, D, H, W]
        max_out = x.amax(dim=1, keepdim=True)           # [B, 1, D, H, W]
        scale = torch.sigmoid(
            self.conv(torch.cat([avg_out, max_out], dim=1))
        )                                               # [B, 1, D, H, W]
        return x * scale


class CBAM3D(nn.Module):
    """Channel Attention → Spatial Attention 串联（与原 CBAM 论文顺序一致）。"""
    def __init__(self, channels: int, reduction: int = 4, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttn3D(channels, reduction)
        self.spatial_attn = SpatialAttn3D(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class DynUNetWithCA(nn.Module):
    """
    DynUNet + CBAM3D Wrapper.

    在 skip_layers(decoder最终32ch特征) → output_block(分类头) 之间
    插入 CBAM3D 注意力，其余部分与原 DynUNet 完全一致。

    对应论文门控思路 (1 + M(x)) * h(x):
      - SpatialAttn3D 生成空间权重图(sigmoid输出)作为 M(x)
      - 当前实现用乘法门控（而非加性），效果等价且更稳定

    兼容性：
      - torch.compile: 全静态shape,可直接编译
      - AMP: 全部 sigmoid/Linear/Conv3d,fp16稳定
      - sliding_window_inference: forward签名不变
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super().__init__()
        self.backbone = build_dynunet(in_channels, out_channels)
        feat_channels = self.backbone.filters[0]  # 是第一层的dynunet的特征通道数
        self.cbam = CBAM3D(channels=feat_channels, reduction=4, spatial_kernel=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.skip_layers(x)    # [B, 32, D, H, W]
        feat = self.cbam(feat)                 # [B, 32, D, H, W]
        return self.backbone.output_block(feat)  # [B, 2, D, H, W]

    def load_backbone_weights(self, path: str, map_location="cpu"):
        """
        从裸 DynUNet checkpoint 加载权重到 backbone。
        处理 _orig_mod. 前缀(torch.compile保存的ckpt)。
        """
        raw = torch.load(path, map_location=map_location)
        src = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        # 去掉 torch.compile 产生的 _orig_mod. 前缀
        src = {k.removeprefix("_orig_mod."): v for k, v in src.items()}
        dst = self.backbone.state_dict()
        matched = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
        dst.update(matched)
        self.backbone.load_state_dict(dst, strict=False)
        print(f"[load_backbone_weights] loaded {len(matched)}/{len(dst)} params from {path}")


def build_dynunet_ca(in_channels: int = 1, out_channels: int = 2) -> DynUNetWithCA:
    return DynUNetWithCA(in_channels, out_channels)
