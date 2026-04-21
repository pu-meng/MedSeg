"""
检查 Task03_Liver 数据集是否完整
用法:  python -m scripts.check.check_dataset --data_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver
"""
import os
import argparse
import glob


def check_dataset(data_root):
    print(f"检查路径: {data_root}")
    print("=" * 50)

    # 1. 检查目录结构
    imagesTr = os.path.join(data_root, "imagesTr")
    labelsTr = os.path.join(data_root, "labelsTr")
    for d in [imagesTr, labelsTr]:
        exists = os.path.isdir(d)
        print(f"{'✅' if exists else '❌'} {d}")

    print()

    # 2. 扫描文件
    images = sorted(glob.glob(os.path.join(imagesTr, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(labelsTr, "*.nii.gz")))
    print(f"image 数量: {len(images)}")
    print(f"label 数量: {len(labels)}")

    # 3. 检查配对
    img_ids = {os.path.basename(f).replace(".nii.gz", "") for f in images}
    lab_ids = {os.path.basename(f).replace(".nii.gz", "") for f in labels}
    only_img = img_ids - lab_ids
    only_lab = lab_ids - img_ids
    if only_img:
        print(f"❌ 有image但没有label: {sorted(only_img)}")
    if only_lab:
        print(f"❌ 有label但没有image: {sorted(only_lab)}")
    if not only_img and not only_lab:
        print("✅ image/label 完全配对")

    # 4. 检查文件大小(过小的可能损坏)
    print()
    print("检查文件大小(小于100KB的文件):")
    bad = []
    for f in images + labels:
        size = os.path.getsize(f)
        if size < 100 * 1024:
            bad.append((f, size))
    if bad:
        for f, s in bad:
            print(f"  ❌ {os.path.basename(f)}  {s/1024:.1f} KB")
    else:
        print("  ✅ 所有文件大小正常")

    # 5. 打印前5个配对样例
    print()
    print("前5个配对样例:")
    for img in images[:5]:
        sid = os.path.basename(img).replace(".nii.gz", "")
        lab = os.path.join(labelsTr, sid + ".nii.gz")
        lab_ok = "✅" if os.path.exists(lab) else "❌"
        img_mb = os.path.getsize(img) / 1024 / 1024
        print(f"  {sid}  image={img_mb:.1f}MB  label={lab_ok}")

    print()
    print("=" * 50)
    print(f"检查完成,共 {len(images)} 个样本")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    args = p.parse_args()
    check_dataset(args.data_root)