"""
calc_patch.py
统计数据集图像尺寸,根据显存推荐合适的 patch size
用法:python calc_patch.py --data_root /path/to/Task03_Liver_0p88mm --vram_gb 11
import glob这里的glob是批量找文件

"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib


def estimate_vram_gb(patch_size, num_classes=3, base_channels=32, num_levels=4):
    """
    估算 UNet3D 训练时的实际显存消耗.

    UNet3D 显存主要来自三部分:
    1. 模型权重本身(相对固定,大概 0.5~1GB)
    2. 前向传播的特征图(每一层都要保留用于跳跃连接)
    3. 反向传播的梯度(和特征图一样大)

    UNet 每层 encoder 特征图大小:
      第0层: patch_size × base_channels
      第1层: patch_size/2 × base_channels*2
      第2层: patch_size/4 × base_channels*4
      ...
    decoder 和 encoder 对称,再乘以 2.
    再加上梯度(×2)和优化器状态 AdamW(×2)
    总系数大概是 ×6.
    num_levels=4默认按照4层encoder/decoder结构估算
    """
    d, h, w = patch_size
    channels = base_channels
    total_voxels = 0

    for level in range(num_levels):
        # 这个是遍历每一层
        current_channels = base_channels * (2**level)
        scale = 2**level
        level_voxels = (d // scale) * (h // scale) * (w // scale) * current_channels

        total_voxels += level_voxels  # encoder
        total_voxels += level_voxels  # decoder (skip connection)
        channels = min(channels * 2, 320)  # nnUNet cap at 320
#设计不大于320,只是nnunet的一个常见的设计习惯
#每个元素按照flozt32算字节数,一个float32=4个字节
    bytes_per_voxel = 4  # float32
    forward_bytes = total_voxels * bytes_per_voxel
    # ×6: 前向 + 反向梯度 + AdamW 两份动量
    total_bytes = forward_bytes * 6
    #这个的乘以6是估算,
    # 加上模型权重本身(固定约 800MB)
    
    total_bytes += 800 * 1024**2
#1KB=1024 B,1MB=1024KB,1GB=1024MB
    return total_bytes / (1024**3)


def calc_patch_size(median_size, vram_gb, num_classes=3):
    """
    从 median_size 出发,逐步缩小直到显存估算 < vram_gb * 0.85
    (留 15% 余量给数据 IO 和其他进程)

    clac_patch_size是已知数据集典型尺寸median_size,和显存大小vram_gb,找一个不爆显存的patch

    """
    budget_gb = vram_gb * 0.85
#这个乘以0.85是留15%余量给数据 IO 和其他进程
    d = int((median_size[0] // 16) * 16)
    h = int((median_size[1] // 16) * 16)
    w = int((median_size[2] // 16) * 16)

    # 不能超过 median_size,只是保险
    d = min(d, int(median_size[0]))
    h = min(h, int(median_size[1]))
    w = min(w, int(median_size[2]))

    # 逐步缩小
    while True:
        est = estimate_vram_gb((d, h, w), num_classes=num_classes)
        if est <= budget_gb:
            break

        # 每次缩小最长的轴,步长 16
        dims = [d, h, w]
        max_ax = int(np.argmax(dims))
        #np.argmax(dims)返回最大值的索引
        dims[max_ax] = max(dims[max_ax] - 16, 16)
        #把最长轴缩小16,但不能低于16,比如[144,224,192]变成[144,208,192]

        d, h, w = dims

        # 防止无限循环
        if d <= 16 and h <= 16 and w <= 16:
            break

    return d, h, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        default="/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver",
    )
    p.add_argument(
        "--vram_gb",
        type=float,
        default=11.0,
        help="你的显卡显存大小(GB),默认 11",
    )
    p.add_argument(
        "--num_classes",
        type=int,
        default=3,
    )
    args = p.parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.data_root, "imagesTr", "*.nii.gz")))
    assert len(img_paths) > 0, f"找不到数据: {args.data_root}"

    print(f"共 {len(img_paths)} 个 case,开始统计图像尺寸...")

    sizes = []

  
    for p in img_paths:
        img = nib.load(p)
        sizes.append(img.shape)
    sizes = np.array(sizes)  # type:ignore
   
#np.median(,axis=0)意思是按照样本维度计算中位值,
#每个case有[D,H,W],分别算D,H,W的中位值
    median_size = np.median(sizes, axis=0)
    min_size = sizes.min(axis=0)
    #sizes.min(axis=0)是得到每个维度的最小值
    max_size = sizes.max(axis=0)

    print("\n========== 图像尺寸统计(预处理后)==========")
    print(
        f"median size : D={median_size[0]:.0f}  H={median_size[1]:.0f}  W={median_size[2]:.0f}"
    )
    print(f"min    size : D={min_size[0]}  H={min_size[1]}  W={min_size[2]}")
    print(f"max    size : D={max_size[0]}  H={max_size[1]}  W={max_size[2]}")

    # 推荐 patch
    patch = calc_patch_size(
        median_size, vram_gb=args.vram_gb, num_classes=args.num_classes
    )
    est = estimate_vram_gb(patch, num_classes=args.num_classes)

    print(f"\n========== 推荐 Patch Size(显存 {args.vram_gb}GB)==========")
    print(f"推荐 patch      : {list(patch)}")
    print(f"估算显存消耗    : {est:.1f} GB  (预算 {args.vram_gb * 0.85:.1f} GB)")
    print("\n在 train.py 启动命令里加上:")
    print(f"  --patch {patch[0]} {patch[1]} {patch[2]}")

    # 不同显存对比
    print("\n========== 不同显存下的推荐 patch ==========")
    for vram in [8, 11, 12, 16, 24, 40]:
        pp = calc_patch_size(median_size, vram_gb=vram, num_classes=args.num_classes)
        est = estimate_vram_gb(pp, num_classes=args.num_classes)
        print(f"  {vram:>3}GB  →  {list(pp)}  (估算 {est:.1f}GB)")


if __name__ == "__main__":
    main()
