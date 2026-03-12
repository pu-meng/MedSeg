# convert_to_nnunet.py 改成软链接版本
"""
convert_to_nnunet.py
os.path.basename(path),取路径最后的文件名
p = "/home/pumengyu/.../imagesTr/liver_001.nii.gz"
os.path.basename(p)  →  "liver_001.nii.gz"

os.symlink(src, dst)
           ↑        ↑
       真实文件   软链接文件（新建的）
p是真实文件的绝对路径,dst是软链接文件的绝对路径
类似Linux命令: ln -s p link
p是真实文件的绝对路径,link是软链接文件的绝对路径


挂载:
mount /dev/sdb1 /mnt/data
#      ↑设备     ↑挂载点（必须是已存在的空目录）
sda是第一个硬盘,sdb是第二个硬盘,sdc是第三个硬盘
sda1是第一个硬盘的第一个分区,sdb1是第二个硬盘的第一个分区,sdc1是第三个硬盘的第一个分区

机械硬盘(HDD):靠着物理旋转读写
固态硬盘(SSD):靠芯片存储

json.dump(obj, fp, indent=2)
obj:要写入的json对象
fp:文件指针,也就是文件 的路径
indent=2:缩进2个空格
"""
import os
import glob
import json


src = "/home/pumengyu/Task03_Liver"
dst = "/home/pumengyu/nnunet/raw/Dataset003_Liver"

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
