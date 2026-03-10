"""
calc_sw_batch_size.py
=====================
根据显存和 patch 大小,推荐最优 sw_batch_size.

nnUNet 思想:
  - overlap 固定 0.5(质量和速度最佳平衡,不需要统计)
  - sw_batch_size 根据显存动态推荐
  - patch_size 已经由数据统计决定

用法:
    python calc_sw_batch_size.py --patch 144 144 144 --gpu_mem_gb 11
    python calc_sw_batch_size.py --patch 144 144 144  # 自动检测显存
"""

import argparse
import math


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patch", type=int, nargs=3, default=[144, 144, 144])
    p.add_argument(
        "--gpu_mem_gb", type=float, default=None, help="显存大小(GB),不传则自动检测"
    )
    p.add_argument(
        "--model", type=str, default="unet3d", choices=["unet3d", "unetr", "swinunetr"]
    )
    p.add_argument(
        "--channels", type=int, default=32, help="UNet3D第一层通道数,默认32"
    )
    return p.parse_args()


def get_gpu_memory():
    try:
        import torch

        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory
            return mem / (1024**3)
    except:
        pass
    return None


def estimate_patch_memory_gb(patch_size, model_name, channels):
    """
    估算单个 patch 推理时的显存占用.
    这是经验公式,不是精确值.

    显存占用来源:
      1. 模型参数(固定,与patch无关)
      2. 推理时的中间激活(与patch大小正比)
      3. 输入输出 tensor
    """
    D, H, W = patch_size
    voxels = D * H * W  # 2,985,984 for 144³

    if model_name == "unet3d":
        # UNet3D: 4个下采样层,特征图逐层减半
        # 最大特征图在第一层: [channels, D, H, W]
        # 粗略估算:约等于 voxels * channels * 4层 * float32(4bytes) * 前后向(2x)
        # 推理只有前向,乘1x
        # channels=(32,64,128,256,320),主要占用在前几层
        activation_mem = voxels * channels * 8 * 4 / (1024**3)  # bytes→GB
    elif model_name in ["unetr", "swinunetr"]:
        # Transformer系列:attention map占用更大
        activation_mem = voxels * channels * 16 * 4 / (1024**3)
    else:
        activation_mem = voxels * channels * 8 * 4 / (1024**3)

    # 加上输入输出:1通道输入 + num_classes输出
    io_mem = voxels * (1 + 3) * 4 / (1024**3)  # float32

    return activation_mem + io_mem


def recommend_sw_batch_size(gpu_mem_gb, patch_size, model_name, channels):
    """
    nnUNet 推荐逻辑:
      可用显存 = 总显存 - 模型参数占用(约1-2GB) - 安全余量(1GB)
      sw_batch_size = floor(可用显存 / 单patch显存)
      但最少为1,最多不超过8(超过8收益递减)
    """
    # 模型参数大概占用
    if model_name == "unet3d":
        model_param_gb = 0.5  # UNet3D参数量约100-200M
    else:
        model_param_gb = 1.5  # UNETR/SwinUNETR更大

    safety_margin = 1.0  # 安全余量

    available_gb = gpu_mem_gb - model_param_gb - safety_margin
    available_gb = max(available_gb, 1.0)

    patch_mem = estimate_patch_memory_gb(patch_size, model_name, channels)

    sw_batch_size = int(math.floor(available_gb / patch_mem))
    sw_batch_size = max(1, min(sw_batch_size, 8))

    return sw_batch_size, patch_mem, available_gb


def main():
    args = parse_args()
    patch_size = tuple(args.patch)

    # 获取显存
    if args.gpu_mem_gb:
        gpu_mem_gb = args.gpu_mem_gb
        print(f"[显存] 手动指定: {gpu_mem_gb} GB")
    else:
        gpu_mem_gb = get_gpu_memory()
        if gpu_mem_gb:
            print(f"[显存] 自动检测: {gpu_mem_gb:.1f} GB")
        else:
            gpu_mem_gb = 11.0
            print(f"[显存] 检测失败,使用默认值: {gpu_mem_gb} GB")

    sw_batch, patch_mem, avail = recommend_sw_batch_size(
        gpu_mem_gb, patch_size, args.model, args.channels
    )

    D, H, W = patch_size
    voxels = D * H * W

    print(
        f"""
{"="*50}
        推理参数推荐(nnUNet思想)
{"="*50}

  模型:          {args.model}
  Patch大小:     {D}×{H}×{W} = {voxels:,} 体素
  GPU显存:       {gpu_mem_gb:.1f} GB
  可用显存:      {avail:.1f} GB(扣除模型参数+安全余量)
  单patch显存:   {patch_mem:.2f} GB(估算)

{"─"*50}
  推荐参数:

  overlap      = 0.5     ← nnUNet固定值,不统计
  sw_batch_size = {sw_batch}     ← 根据显存推算

{"─"*50}
  overlap 取值说明:

  overlap=0.25  快,但边缘预测差,tumor dice损失明显
  overlap=0.50  nnUNet默认,质量/速度最优平衡  ✅
  overlap=0.75  最准,但推理时间是0.5的 ~4倍,通常不值得

{"─"*50}
  在 train.py 中的调用:

  metrics = validate_sliding_window(
      model, val_loader, device,
      roi_size={patch_size},
      sw_batch_size={sw_batch},   # ← 用这个
      overlap=0.5,                 # ← 固定0.5
      num_classes=3,
  )
{"="*50}

  💡 如果推理时OOM(显存不足):
     把 sw_batch_size 从 {sw_batch} 改为 {max(1, sw_batch-1)}

  💡 如果推理很慢但显存有余:
     sw_batch_size 可以适当增大,最大建议不超过 8
"""
    )


if __name__ == "__main__":
    main()
