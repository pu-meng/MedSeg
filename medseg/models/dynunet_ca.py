import torch
import torch.nn as nn
from monai.networks.nets.dynunet import DynUNetSkipLayer
from medseg.models.dynunet import build_dynunet


# ---------------------------------------------------------------------------
# 改动一:Slice-wise 2D 分支(对应论文 Figure 5 的 2D U-Net 分支)
# 用 Conv3d(kernel=(1,3,3)) 模拟 slice-wise 2D 卷积,无需 reshape
# 只加在最浅层(32ch),深层分辨率低收益边际化
# ---------------------------------------------------------------------------
class SliceWise2DBranch(nn.Module):
    """
    论文 DCUNet-Tumor Figure 5A:2D 分支提取 slice-level 平面特征。
    实现:Conv3d(kernel=(1,3,3)) 等价于对每个 axial slice 做 2D Conv,
    不需要 reshape,对 sliding_window_inference 友好。
    融合:concat(3D feat, 2D feat) → Conv1×1×1 压回原通道(论文 feature fusion)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv3d(
                channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
            ),  # (D,H,W)这个kernel_size是故意的在,D维度不跨slice,效果等价于对每个slice单独做2D卷积
            nn.InstanceNorm3d(
                channels, affine=True
            ),  # affine=true,表示在归一化后加上可学习的仿射变换(scale和shift),增加模型表达能力
            nn.LeakyReLU(
                0.01, inplace=True
            ),  # inplace=true,表示直接在输入上进行修改,节省内存,但会改变输入数据,需要注意是否会影响后续使用
            nn.Conv3d(
                channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
            ),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        # concat(3D, 2D) 意思是self.fuse的输入是torch.cat([x, feat_2d], dim=1),所以输入通道数是channels * 2
        # 然后kernel_size=1只混合通道,不改变空间尺寸,空间D/H/W不变
        # affine=True表示在InstanceNorm3d后加上一个线性变换y=\alpha *x+\beta,这里的\alpha和\beta都是可以学习的参数
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, D, H, W]
        feat_2d: [B, C, D, H, W](等价于对每个 slice 做 2D 卷积)
        输出:融合后的特征,shape 与 x 相同
        这里的x就代表是3D特征;feat_2d代表是2D特征,通过conv2d提取的slice-wise 2D特征;

        """
        feat_2d = self.conv2d(x)
        return self.fuse(torch.cat([x, feat_2d], dim=1))


# ---------------------------------------------------------------------------
# 改动二:Attention Gate(对应论文 Figure 6 + Eq.3)
# 论文公式:A(x) = (1 + M(x)) × T(x)
#   T(x) = skip feature(encoder 输出)
#   M(x) = soft attention map(sigmoid,[0,1]),由 skip + gating signal 生成
#   (1 + M(x)):残差注意力,M=0 时退化为原始特征,不会比原来差
# ---------------------------------------------------------------------------
class AttGate3D(nn.Module):
    """
    论文 Soft Attention Gate(残差形式,Eq.3)。
    x: skip feature  [B, C, D, H, W]
    g: gating signal(来自更深层 decoder 上采样后)[B, C_g, D, H, W]
    输出:(1 + alpha) * x,shape 与 x 相同
    """

    def __init__(self, in_channels: int, gate_channels: int):
        super().__init__()
        mid = max(1, in_channels // 2)
        self.theta_x = nn.Conv3d(in_channels, mid, kernel_size=1, bias=False)
        self.phi_g = nn.Conv3d(gate_channels, mid, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv3d(mid, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        # 如果输入通道C_in,到输出通道C_in,kernel=1,则卷积核形状是[C_out, C_in, 1, 1, 1]
        # kernel=1是不会聚合周围位置,但是依旧会混合通道

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """ """
        # g 的空间尺寸可能与 x 不同(来自更深层上采样),对齐到 x
        if g.shape[2:] != x.shape[2:]:
            g = nn.functional.interpolate(
                g, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        tx = self.theta_x(x)  # [B,C,D,H,W] -> [B, mid, D, H, W]
        pg = self.phi_g(g)  # [B,C,D,H,W] -> [B, mid, D, H, W]
        attn = torch.relu(tx + pg)
        alpha = self.psi(attn)  # [B, 1, D, H, W],值域 [0,1]
        return (1.0 + alpha) * x  # 论文 Eq.3 残差注意力


# ---------------------------------------------------------------------------
# 改动三:子类化 DynUNetSkipLayer,注入 AttGate + 可选 2D 分支
# ---------------------------------------------------------------------------
class SkipLayerWithAttn(DynUNetSkipLayer):
    """
    MONAI源代码(monai/networks/nets/dynunet.py)
    |
    |
    ----DynUNetSkipLayer   #负责跳跃连接的递归前向传播逻辑
    |
    |
    |
    |
    ----SkipLayerWithAttn(DynUNetSkipLayer)  #自定义的子类:重写forward,加入注意力门控
    """

    def __init__(
        self,
        index,
        downsample,
        upsample,
        next_layer,
        att_gate: nn.Module,
        slice2d: nn.Module = None,
        heads=None,
        super_head=None,
    ):
        # super().__init__()调用父类DynUNetSkipLayer的构造函数,里面的参数是传递给这个父类的
        super().__init__(index, downsample, upsample, next_layer, heads, super_head)
        self.att_gate = att_gate
        self.slice2d = slice2d  # att_gate和slice2d是神经网络模块对象
        # att_gate是AttGate3D()这个网络模块
        # slice2d是SliceWise2DBranch()这个网络模块

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        downout = self.downsample(x)  # skip feature
        nextout = self.next_layer(downout)  # 更深层 decoder 输出(含上采样)

        # 2D 分支融合(只有 L1)
        if self.slice2d is not None:
            downout = self.slice2d(downout)

        # Attention Gate(论文 Eq.3 残差注意力)
        downout = self.att_gate(downout, g=nextout)

        upout = self.upsample(nextout, downout)
        if self.super_head is not None and self.heads is not None and self.index > 0:
            self.heads[self.index - 1] = self.super_head(upout)
        return upout


# 在DyUNet中,self.heads这个列表对象在所有的SkipLayer之间共享一个引用
# self.heads存储了多个不同尺度的预测结果,训练时候每个都算loss
# index=1存到heads[0]
# index=2存到heads[1]
# self.super_head is not None 模型开启了深监督,
# self.heads is not None 存储辅助输出的列表存在
# self.index > 0 不是最浅层(最浅层index=0没有 skip 连接,不需要深监督)


# ---------------------------------------------------------------------------
# 主模型:DynUNetWithCA(实验10)
# ---------------------------------------------------------------------------
class DynUNetWithCA(nn.Module):
    """
        实验10:DynUNet + 2D/3D 特征融合 + Skip Attention Gate

        改动:
        1. L1 skip(32ch)加 SliceWise2DBranch:融合 slice-wise 2D 平面特征
        2. 全部 4 个 skip(32/64/128/256ch)加 AttGate3D(论文 Eq.3 残差注意力)
        3. bottleneck(320ch)不改动

    DynUNetWithCA
      │
      ├── backbone: DynUNet(MONAI原版,build_dynunet构建)
      │   └── skip_layers: 原本是 DynUNetSkipLayer(递归嵌套)
      │                        ↓
                        被 _replace_skip_layers 替换掉
      │
      └── skip_layers 替换为 SkipLayerWithAttn(递归嵌套,4层)
          │
          ├── att_gate: AttGate3D        ← 每层都有,共4个
          │
          └── slice2d: SliceWise2DBranch ← 只有L1(index=0)有,其余为None

      - DynUNetWithCA 是组装者,把backbone里的原始skip层换成增强版的
      - SkipLayerWithAttn 是增强版skip层,在原来逻辑上插入了注意力和2D分支
      - AttGate3D 和 SliceWise2DBranch 是功能模块,被SkipLayerWithAttn调用
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super().__init__()
        self.backbone = build_dynunet(in_channels, out_channels, deep_supervision=False)
        filters = self.backbone.filters  # [32, 64, 128, 256, 320]
        # filters是每层的通道数,对应UNet的5层,bottleneck:320ch是最深层
        # 构建 attention gates
        # skip_ch = filters[i],gate_ch = next layer 输出通道
        # next_layer 输出 = decoder upout,通道数 = filters[i+1](更深层 skip 通道)
        # 但 L3 的 next_layer 是 bottleneck,输出 filters[4]=320
        # 所以 gate_channels = filters[i+1] for i in 0..3
        att_gates = nn.ModuleList(
            [AttGate3D(filters[i], filters[i + 1]) for i in range(4)]
        )
        # nn.ModuleList是能被PyTorch感知的列表,存放多个nn.Module,用普通的列表[],PyTorch不会把里面的参数纳入model.parameters()
        # 训练时候这些参数不会更新

        # 只有 L1(index=0,32ch)加 2D 分支
        slice2d_l1 = SliceWise2DBranch(filters[0])

        # 递归替换 skip_layers
        self.backbone.skip_layers = self._replace_skip_layers(
            self.backbone.skip_layers, att_gates, slice2d_l1
        )

    def _replace_skip_layers(self, layer, att_gates, slice2d_l1):
        """把原本的 DynUNetSkipLayer 替换为→ SkipLayerWithAttn"""
        from monai.networks.nets.dynunet import DynUNetSkipLayer

        if not isinstance(layer, DynUNetSkipLayer):
            return (
                layer  # 遇到bottleneck就会中止,因为bottleneck是普通的nn.Module(卷积块)
            )

        idx = layer.index  # .index是父类的属性
        # 先递归替换更深层
        new_next = self._replace_skip_layers(layer.next_layer, att_gates, slice2d_l1)
        #先递归

        return SkipLayerWithAttn(
            index=idx,
            downsample=layer.downsample,
            upsample=layer.upsample,
            next_layer=new_next,
            att_gate=att_gates[idx],
            slice2d=slice2d_l1 if idx == 0 else None,
            heads=layer.heads,
            super_head=layer.super_head,
        )
#递归就是函数调用自己,必须有终止条件,
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.skip_layers(x)  # encoder+decoder(含bottleneck)
        out = self.backbone.output_block(feat)  # 最终输出 [B, C, D, H, W]
        # 深监督：训练时 heads 存有各辅助尺度输出，上采样到主输出尺寸后拼成 [B, N, C, D, H, W]
        if self.training and self.backbone.deep_supervision:
            target_size = out.shape[2:]
            heads = [out] + [
                nn.functional.interpolate(h, size=target_size, mode="trilinear", align_corners=False)
                for h in self.backbone.heads if h is not None
            ]
            return torch.stack(heads, dim=1)
        return out

    def load_backbone_weights(self, path: str, map_location="cpu"):
        """
        从裸 DynUNet checkpoint 加载权重到 backbone。
        新增的 att_gate / slice2d 参数随机初始化(不在 src 里)。
        处理 _orig_mod. 前缀(torch.compile 保存的 ckpt)。
        """
        raw = torch.load(path, map_location=map_location, weights_only=False)
        src = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        # 去掉 torch.compile 前缀,再加上 backbone. 前缀以匹配当前 state_dict
        src = {"backbone." + k.removeprefix("_orig_mod."): v for k, v in src.items()}
        #这个self.state_dict()是Pytorch的nn.Module的方法,
        #格式为{参数名:tensor},dst是当前模型,包含backbone+其他新增层的完整的state_dict
        dst = self.state_dict()
        #src=从问价加载的旧的checkpoint
        matched = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
        dst.update(matched)
        self.load_state_dict(dst, strict=False)
        print(
            f"[load_backbone_weights] loaded {len(matched)}/{len(dst)} params from {path}"
        )


def build_dynunet_ca(in_channels: int = 1, out_channels: int = 2) -> DynUNetWithCA:
    return DynUNetWithCA(in_channels, out_channels)

#继nn.Module就可以使用nn.Module的self不需要自己规定
#比如方法:self.state_dict(),self.load_state_dict(),self.parameters(),self.to(device),self.eval(),self.train()等方法都是nn.Module提供的
#比如属性,不需要():self.training,self.device,self._parameters,self._buffers,self._modules等属性也是nn.Module提供的