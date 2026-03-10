"""
tools/preprocess_stage2.py
stage 1的任务是找到肝脏在哪里,裁出liver ROI
这个是stage2的任务是找到肿瘤在哪里；
功能: 基于 GT liver mask, 从原始CT中裁出 liver ROI,
      重新映射 label (肿瘤=1, 其余=0), 保存为 .nii.gz

用法:
    python -m tools.preprocess_stage2 \
        --data_root /home/pumengyu/Task03_Liver \
        --output_dir /home/pumengyu/stage2_crops \
        --margin 20
SimpleITK是专门用来读写医学影像数据的库(.nii.gz)等，类似
img_sitk = sitk.ReadImage("liver_001.nii.gz")
img_sitk会保存像素值,每个体素的物理尺寸,以及图像的坐标系等信息

"""

import os
import argparse
import glob
import numpy as np
import SimpleITK as sitk


def crop_liver_roi(image_arr, label_arr, margin=20):
    """
    根据 liver mask (label>=1) 裁剪 bounding box + margin
    返回: cropped_image, stage2_label
    stage2_label: 0=背景+肝脏实质, 1=肿瘤;
    coords[0]=array([12,12,...])所有肝脏体素的z坐标
    np.where(liver_mask)只找liver_mask为True的坐标;
    coords[0][i]是第i个True的z坐标,
    三维的体素排序是,先固定z,在z平面内按行优先顺序遍历y,x;
    margin=20 就是在z,y,x三个维度上都扩展20个体素,注意要保证不越界.
    """
    liver_mask = label_arr >= 1  # 肝脏+肿瘤都算前景

    coords = np.where(liver_mask)
    if len(coords[0]) == 0:
        print("  [warn] 无肝脏区域, 跳过")
        return None, None

    # bounding box + margin
    mins = [max(0, c.min() - margin) for c in coords]
    maxs = [min(s - 1, c.max() + margin) for c, s in zip(coords, image_arr.shape)]

    z0, y0, x0 = mins
    z1, y1, x1 = [m + 1 for m in maxs]

    cropped_image = image_arr[z0:z1, y0:y1, x0:x1]
    cropped_label = label_arr[z0:z1, y0:y1, x0:x1]

    # label 重映射: 肿瘤(2)->1, 其余->0
    stage2_label = np.zeros_like(cropped_label)
    stage2_label[cropped_label == 2] = 1

    return cropped_image, stage2_label


def save_nii(arr, ref_sitk, out_path):
    out_sitk = sitk.GetImageFromArray(arr)
    out_sitk.SetSpacing(ref_sitk.GetSpacing())
    out_sitk.SetOrigin(ref_sitk.GetOrigin())
    out_sitk.SetDirection(ref_sitk.GetDirection())
    sitk.WriteImage(out_sitk, out_path)


def process_one(image_path, label_path, out_img_dir, out_lab_dir, margin, sid):
    print(f"处理: {sid}")

    img_sitk = sitk.ReadImage(image_path)
    lab_sitk = sitk.ReadImage(label_path)

    img_arr = sitk.GetArrayFromImage(img_sitk)  # shape: (Z, Y, X)
    lab_arr = sitk.GetArrayFromImage(lab_sitk)

    cropped_img, stage2_lab = crop_liver_roi(img_arr, lab_arr, margin)
    if cropped_img is None:
        return False

    tumor_vox = int((stage2_lab == 1).sum())
    print(f"  ROI shape={cropped_img.shape}  tumor_vox={tumor_vox}")

    save_nii(cropped_img, img_sitk, os.path.join(out_img_dir, f"{sid}.nii.gz"))
    save_nii(
        stage2_lab.astype(np.uint8),
        lab_sitk,
        os.path.join(out_lab_dir, f"{sid}.nii.gz"),
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--margin", type=int, default=20)
    args = parser.parse_args()

    images_dir = os.path.join(args.data_root, "imagesTr")
    labels_dir = os.path.join(args.data_root, "labelsTr")

    out_img_dir = os.path.join(args.output_dir, "imagesTr")
    out_lab_dir = os.path.join(args.output_dir, "labelsTr")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lab_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

    assert len(images) == len(labels) and len(images) > 0, (
        f"数据有问题: images={len(images)}, labels={len(labels)}"
    )

    print(f"共 {len(images)} 个样本, margin={args.margin}")

    ok, skip = 0, 0
    for img_path, lab_path in zip(images, labels):
        sid = os.path.basename(img_path).replace(".nii.gz", "")
        success = process_one(
            img_path, lab_path, out_img_dir, out_lab_dir, args.margin, sid
        )
        if success:
            ok += 1
        else:
            skip += 1

    print(f"\n完成: 成功={ok}, 跳过={skip}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
