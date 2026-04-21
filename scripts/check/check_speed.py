
import time
import  glob
from medseg.data.dataset_offline import OfflineDataset
from medseg.data.transforms_offline import build_train_transforms
from torch.utils.data import DataLoader

pt_dir = "/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt"
paths = sorted(glob.glob(pt_dir + "/*.pt"))[:20]

tf = build_train_transforms((96, 96, 96))
ds = OfflineDataset(paths, transform=tf, repeats=1)

loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)

t0 = time.time()
for i, batch in enumerate(loader):
    t1 = time.time()
    print(f"batch {i}: {t1-t0:.2f}s")
    t0 = t1
    if i >= 5:
        break
