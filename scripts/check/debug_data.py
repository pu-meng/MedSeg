from medseg.data.msd import load_msd_dataset, fixed_split
from medseg.data.transforms import build_train_transforms
import nibabel as nib

DATA_ROOT = "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver"

items, _ = load_msd_dataset(DATA_ROOT)
tr, va = fixed_split(items, val_ratio=0.2, seed=0)

tfm = build_train_transforms(patch_size=(96, 96, 96))

# 1) 随机抽 3 个病例,看原始 label 的 unique
print("==== raw label unique check ====")
for i in [0, 1, 2]:
    sample = tr[i]

    lab = nib.load(sample["label"]).get_fdata()
    u = sorted(set(lab.flatten().tolist()))
    u = u[:20]
    print(sample["id"], "unique(head):", u)

print("\n==== after transform check (patch) ====")
for i in [0, 1, 2]:
    out = tfm(tr[i])  # 可能返回 list[dict](num_samples>1),也可能返回 dict

    outs = out if isinstance(out, list) else [out]

    for j, o in enumerate(outs):
        y = o["label"]
        if y.ndim == 4:
            y = y[0]  # [1, Z, Y, X] -> [Z, Y, X]
        u = y.unique().cpu().tolist()
        tumor = int((y == 2).sum().item())
        liver = int((y == 1).sum().item())
        print(
            f"{tr[i]['id']} patch{j}: unique={u}, liver_vox={liver}, tumor_vox={tumor}, has_tumor={tumor > 0}"
        )
