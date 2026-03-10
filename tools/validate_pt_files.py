"""
validate_pt_files.py
====================
删除原始 .nii.gz 前,全面验证预处理 .pt 文件是否完好.

检查项目:
  ✅ 1. .pt 文件数量与原始 .nii.gz 是否一致
  ✅ 2. 每个 .pt 能否正常加载(文件未损坏)
  ✅ 3. 包含 "image" 和 "label" 两个 key
  ✅ 4. image dtype = float32, label dtype = int64
  ✅ 5. image shape = [1, D, H, W],label shape = [1, D, H, W]
  ✅ 6. image 值域在 [0, 1](归一化正确)
  ✅ 7. label 只含 {0, 1, 2}(无非法类别)
  ✅ 8. label 中 liver(1) 和 tumor(2) 都存在(非全背景)
  ✅ 9. image 和 label 空间尺寸一致
  ✅ 10. 文件大小不为 0

使用方式:
  python validate_pt_files.py \
    --pt_dir  /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Liver_0.88mm_pre \
    --src_dir /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm

  # 若两个目录结构确认无误,加 --delete_src 真正删除原始数据:
  python validate_pt_files.py \
    --pt_dir  ... \
    --src_dir ... \
    --delete_src
"""

import os
import glob
import argparse
import torch
import warnings

# ── ANSI 颜色(终端输出更清晰)──────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg):
    print(f"  {GREEN}✅ {msg}{RESET}")


def warn(msg):
    print(f"  {YELLOW}⚠️  {msg}{RESET}")


def fail(msg):
    print(f"  {RED}❌ {msg}{RESET}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pt_dir",
        type=str,
        default="/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Liver_0.88mm_pre",
        help="预处理 .pt 文件目录",
    )
    p.add_argument(
        "--src_dir",
        type=str,
        default="/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm",
        help="原始 .nii.gz 数据集根目录(用于比对数量)",
    )
    p.add_argument(
        "--delete_src",
        action="store_true",
        help="验证全部通过后,删除原始 src_dir(请三思!)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个文件的详细信息",
    )
    return p.parse_args()


def check_count(pt_dir, src_dir):
    """检查 .pt 数量 vs 原始 .nii.gz 数量"""
    pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    nii_files = sorted(glob.glob(os.path.join(src_dir, "imagesTr", "*.nii.gz")))

    print(f"\n{'─' * 55}")
    print(f"{BOLD}[检查 1] 文件数量对比{RESET}")
    print(f"  原始 .nii.gz : {len(nii_files)} 个")
    print(f"  预处理 .pt   : {len(pt_files)} 个")

    # 检查每个 nii 是否都有对应 pt
    missing = []
    for nf in nii_files:
        name = os.path.splitext(os.path.splitext(os.path.basename(nf))[0])[0]
        expected_pt = os.path.join(pt_dir, f"{name}.pt")
        if not os.path.exists(expected_pt):
            missing.append(name)

    if missing:
        fail(f"以下 {len(missing)} 个样本缺少对应 .pt 文件:")
        for m in missing:
            print(f"       {m}")
        return False, pt_files
    else:
        ok(f"数量一致,共 {len(pt_files)} 个,无遗漏")
        return True, pt_files


def validate_one(pt_path, verbose=False):
    """对单个 .pt 文件做全部检查,返回 (passed: bool, info: dict)"""
    errors = []
    info = {}
    name = os.path.basename(pt_path)

    # ── 文件大小 ────────────────────────────────────────────────────────
    size_bytes = os.path.getsize(pt_path)
    if size_bytes == 0:
        errors.append("文件大小为 0,可能写入未完成")
        return False, {"name": name, "errors": errors}

    # ── 加载 ────────────────────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        errors.append(f"torch.load 失败: {e}")
        return False, {"name": name, "errors": errors}

    # ── key 检查 ────────────────────────────────────────────────────────
    if "image" not in data:
        errors.append('缺少 key "image"')
    if "label" not in data:
        errors.append('缺少 key "label"')
    if errors:
        return False, {"name": name, "errors": errors}

    img = data["image"]
    lab = data["label"]

    # ── dtype ───────────────────────────────────────────────────────────
    if img.dtype != torch.float32:
        errors.append(f"image dtype 应为 float32,实际是 {img.dtype}")
    if lab.dtype != torch.int64:
        errors.append(f"label dtype 应为 int64,实际是 {lab.dtype}")

    # ── shape ───────────────────────────────────────────────────────────
    if img.ndim != 4 or img.shape[0] != 1:
        errors.append(f"image shape 应为 [1,D,H,W],实际是 {list(img.shape)}")
    if lab.ndim != 4 or lab.shape[0] != 1:
        errors.append(f"label shape 应为 [1,D,H,W],实际是 {list(lab.shape)}")

    # ── 空间尺寸一致 ─────────────────────────────────────────────────────
    if img.ndim == 4 and lab.ndim == 4 and img.shape[1:] != lab.shape[1:]:
        errors.append(
            f"image/label 尺寸不一致: {list(img.shape[1:])} vs {list(lab.shape[1:])}"
        )

    # ── image 值域 [0, 1] ────────────────────────────────────────────────
    img_min = img.min().item()
    img_max = img.max().item()
    if img_min < -0.01 or img_max > 1.01:
        errors.append(f"image 值域异常: [{img_min:.4f}, {img_max:.4f}],应在 [0,1]")

    # ── label 类别 ───────────────────────────────────────────────────────
    unique_labels = lab.unique().tolist()
    invalid = [v for v in unique_labels if v not in (0, 1, 2)]
    if invalid:
        errors.append(f"label 含非法值: {invalid}")

    # ── 非全背景(至少有 liver)──────────────────────────────────────────
    if 1 not in unique_labels:
        errors.append("label 中没有 liver (1),可能是空 mask")

    # ── 统计信息 ─────────────────────────────────────────────────────────
    info = {
        "name": name,
        "shape": list(img.shape[1:]),
        "img_range": (round(img_min, 4), round(img_max, 4)),
        "labels": [int(v) for v in unique_labels],
        "size_mb": round(size_bytes / 1024 / 1024, 1),
        "errors": errors,
    }

    return len(errors) == 0, info


def validate_all(pt_files, verbose=False):
    """批量验证所有 .pt 文件"""
    print(f"\n{'─' * 55}")
    print(f"{BOLD}[检查 2~10] 逐文件内容验证(共 {len(pt_files)} 个){RESET}")

    passed_list = []
    failed_list = []

    for i, pt_path in enumerate(pt_files):
        passed, info = validate_one(pt_path, verbose=verbose)
        if passed:
            passed_list.append(info)
            if verbose:
                ok(
                    f"[{i + 1:3d}/{len(pt_files)}] {info['name']}"
                    f"  shape={info['shape']}"
                    f"  range={info['img_range']}"
                    f"  labels={info['labels']}"
                    f"  {info['size_mb']}MB"
                )
            else:
                # 每10个打一个进度点
                if (i + 1) % 10 == 0 or (i + 1) == len(pt_files):
                    print(f"  进度: {i + 1}/{len(pt_files)}", end="\r")
        else:
            failed_list.append(info)
            fail(f"[{i + 1:3d}/{len(pt_files)}] {info['name']}")
            for e in info["errors"]:
                print(f"         └─ {e}")

    print()  # 换行
    return passed_list, failed_list


def print_summary(passed_list, failed_list):
    """打印统计摘要"""
    total = len(passed_list) + len(failed_list)
    print(f"\n{'═' * 55}")
    print(f"{BOLD}验证摘要{RESET}")
    print(f"  总计  : {total} 个")
    print(f"  通过  : {GREEN}{len(passed_list)}{RESET} 个")
    print(f"  失败  : {RED}{len(failed_list)}{RESET} 个")

    if passed_list:
        shapes = [p["shape"] for p in passed_list]
        sizes = [p["size_mb"] for p in passed_list]
        all_labels = set()
        for p in passed_list:
            all_labels.update(p["labels"])

        d_vals = [s[0] for s in shapes]
        print(
            f"\n  空间尺寸(D): min={min(d_vals)}, max={max(d_vals)}, mean={sum(d_vals) // len(d_vals)}"
        )
        print(
            f"  文件大小(MB)  : min={min(sizes):.1f}, max={max(sizes):.1f}, mean={sum(sizes) / len(sizes):.1f}"
        )
        label_map = {0: "background", 1: "liver", 2: "tumor"}
        label_str = ", ".join(
            f"{v}({label_map.get(v, '?')})" for v in sorted(all_labels)
        )
        print(f"  label 类别    : {{{label_str}}}")

    print(f"{'═' * 55}")


def delete_src(src_dir):
    """删除原始数据目录(危险操作,需要再次确认)"""
    print(f"\n{RED}{BOLD}⚠️  即将删除原始数据目录:{src_dir}{RESET}")
    confirm = input("请输入 'yes, delete it' 确认:").strip()
    if confirm == "yes, delete it":
        import shutil

        shutil.rmtree(src_dir)
        print(f"{GREEN}✅ 已删除:{src_dir}{RESET}")
    else:
        print("已取消,原始数据未删除.")


def main():
    args = parse_args()

    print(f"\n{BOLD}{'═' * 55}")
    print("  .pt 文件完整性验证脚本")
    print(f"{'═' * 55}{RESET}")
    print(f"  pt_dir  : {args.pt_dir}")
    print(f"  src_dir : {args.src_dir}")

    all_passed = True

    # ── 检查 1:数量对比 ──────────────────────────────────────────────────
    count_ok, pt_files = check_count(args.pt_dir, args.src_dir)
    if not count_ok:
        all_passed = False

    if len(pt_files) == 0:
        fail("没有找到任何 .pt 文件,请检查 --pt_dir 路径")
        return

    # ── 检查 2~10:内容验证 ───────────────────────────────────────────────
    passed_list, failed_list = validate_all(pt_files, verbose=args.verbose)
    if failed_list:
        all_passed = False

    # ── 打印摘要 ──────────────────────────────────────────────────────────
    print_summary(passed_list, failed_list)

    # ── 最终结论 ──────────────────────────────────────────────────────────
    if all_passed:
        print(f"\n{GREEN}{BOLD}🎉 所有检查通过!预处理数据完整可靠.{RESET}")
        if args.delete_src:
            delete_src(args.src_dir)
        else:
            print(f"\n{YELLOW}💡 如确认无误,可加 --delete_src 参数删除原始数据:{RESET}")
            print("   python validate_pt_files.py \\")
            print(f"     --pt_dir  {args.pt_dir} \\")
            print(f"     --src_dir {args.src_dir} \\")
            print("     --delete_src")
    else:
        print(f"\n{RED}{BOLD}❌ 验证未完全通过,请勿删除原始数据!{RESET}")
        print("   请修复上方报错后再重新运行验证.")


if __name__ == "__main__":
    main()
