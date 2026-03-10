# convert_to_nnunet.py 改成软链接版本
import os, glob, json

src = "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm"
dst = "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/nnunet/raw/Dataset003_Liver"

os.makedirs(f"{dst}/imagesTr", exist_ok=True)
os.makedirs(f"{dst}/labelsTr", exist_ok=True)

# 软链接image(不占磁盘空间)
for p in sorted(glob.glob(f"{src}/imagesTr/*.nii.gz")):
    name = os.path.basename(p).replace(".nii.gz", "")
    link = f"{dst}/imagesTr/{name}_0000.nii.gz"
    if not os.path.exists(link):
        os.symlink(p, link)
    print(f"linked: {name}")

# 软链接label
for p in sorted(glob.glob(f"{src}/labelsTr/*.nii.gz")):
    name = os.path.basename(p)
    link = f"{dst}/labelsTr/{name}"
    if not os.path.exists(link):
        os.symlink(p, link)
    print(f"linked: {name}")

# dataset.json
dataset = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "liver": 1, "tumor": 2},
    "numTraining": len(glob.glob(f"{src}/imagesTr/*.nii.gz")),
    "file_ending": ".nii.gz",
}
with open(f"{dst}/dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
print("完成,使用软链接,不占额外空间")
