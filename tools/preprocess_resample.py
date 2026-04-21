import os
import glob
import numpy as np
import nibabel as nib

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
)
from monai.data import Dataset, DataLoader


def main():
    src = "/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver"
    dst = "/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_0.88mm"
    pixdim = (0.88, 0.88, 0.88)

    os.makedirs(os.path.join(dst, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(dst, "labelsTr"), exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(src, "imagesTr", "*.nii.gz")))
    lab_paths = sorted(glob.glob(os.path.join(src, "labelsTr", "*.nii.gz")))
    assert len(img_paths) == len(lab_paths), "imagesTr/labelsTr count mismatch"

    items = []
    for ip in img_paths:
        name = os.path.basename(ip)
        lp = os.path.join(src, "labelsTr", name)
        if not os.path.exists(lp):
            raise FileNotFoundError(lp)
        out_i = os.path.join(dst, "imagesTr", name)
        out_l = os.path.join(dst, "labelsTr", name)
        if os.path.exists(out_i) and os.path.exists(out_l):
            continue
        items.append({"image": ip, "label": lp, "name": name})

    print(f"To process: {len(items)} cases")

    tfm = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")
            ),
        ]
    )

    ds = Dataset(items, transform=tfm)
    # num_workers=0 最稳,先保证跑通;跑通后你再改成 2/4 加速
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for b in loader:
        name = b["name"][0]
        img = b["image"][0, 0].cpu().numpy().astype(np.float32)  # [D,H,W]
        lab = b["label"][0, 0].cpu().numpy().astype(np.uint8)  # [D,H,W]

        out_i = os.path.join(dst, "imagesTr", name)
        out_l = os.path.join(dst, "labelsTr", name)

        # 用原 affine(或 identity),离线训练一般不靠 affine;你如果很在意几何一致性再细化
        nib.save(nib.Nifti1Image(img, np.eye(4)), out_i)
        nib.save(nib.Nifti1Image(lab, np.eye(4)), out_l)

        print("saved:", name)


if __name__ == "__main__":
    main()
