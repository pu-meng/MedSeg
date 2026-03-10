"""
check_pt_labels.py
==================
快速验证 .pt 文件的 label 是否正确(值是否为 0/1/2,shape是否对).
运行前请修改 PT_DIR 为你的预处理目录.

用法:
    python check_pt_labels.py --pt_dir /path/to/preprocessed_pt
    python check_pt_labels.py --pt_dir /path/to/preprocessed_pt --n 5  # 只检查前5个
"""

import os
import glob
import argparse
import torch



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pt_dir", type=str, required=True)
    p.add_argument("--n", type=int, default=0, help="只检查前N个文件,0=全部")
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.pt_dir, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"找不到 .pt 文件: {args.pt_dir}")
    if args.n > 0:
        paths = paths[:args.n]

    print(f"共检查 {len(paths)} 个 .pt 文件\n")

    errors = []
    all_unique_labels = set()

    for i, pt_path in enumerate(paths):
        name = os.path.basename(pt_path)
        data = torch.load(pt_path, map_location="cpu", weights_only=False)

        img = data["image"]
        lab = data["label"]

        # shape检查
        img_shape = tuple(img.shape)
        lab_shape = tuple(lab.shape)

        # label值检查
        unique_vals = sorted(lab.unique().tolist())
        all_unique_labels.update(unique_vals)

        # image值范围检查(应该在[0,1]之间,因为已经归一化)
        img_min = float(img.min())
        img_max = float(img.max())

        # dtype检查
        img_dtype = str(img.dtype)
        lab_dtype = str(lab.dtype)

        # 体素统计
        lab_np = lab.numpy().flatten()
        has_liver = int((lab_np == 1).sum())
        has_tumor = int((lab_np == 2).sum())
        total = lab_np.size

        # 判断是否有问题
        ok = True
        issues = []

        if set(unique_vals) - {0, 1, 2}:
            issues.append(f"⚠️ label有异常值: {unique_vals}")
            ok = False

        if img_min < -0.01 or img_max > 1.01:
            issues.append(f"⚠️ image值范围异常: [{img_min:.3f}, {img_max:.3f}](应在[0,1])")
            ok = False

        if img.dtype != torch.float32:
            issues.append(f"⚠️ image dtype={img_dtype}(应为float32)")
            ok = False

        if lab.dtype not in [torch.int64, torch.int32]:
            issues.append(f"⚠️ label dtype={lab_dtype}(应为int64)")
            ok = False

        if img_shape[1:] != lab_shape[1:]:
            issues.append(f"⚠️ image和label空间尺寸不一致: {img_shape} vs {lab_shape}")
            ok = False

        status = "✅" if ok else "❌"
        print(f"[{i+1:03d}] {status} {name}")
        print(f"       image: shape={img_shape}, dtype={img_dtype}, range=[{img_min:.3f},{img_max:.3f}]")
        print(f"       label: shape={lab_shape}, dtype={lab_dtype}, unique={unique_vals}")
        print(f"       体素:  liver={has_liver:,} ({has_liver/total*100:.2f}%)  "
              f"tumor={has_tumor:,} ({has_tumor/total*100:.2f}%)")
        if issues:
            for issue in issues:
                print(f"       {issue}")
            errors.append(name)
        print()

    print("="*55)
    print(f"检查完毕:{len(paths)} 个文件,{len(errors)} 个有问题")
    print(f"所有文件出现过的 label 值: {sorted(all_unique_labels)}")
    if errors:
        print("\n❌ 有问题的文件:")
        for e in errors:
            print(f"   {e}")
    else:
        print("\n✅ 全部正常,预处理数据没有问题!")
    print("="*55)


if __name__ == "__main__":
    main()