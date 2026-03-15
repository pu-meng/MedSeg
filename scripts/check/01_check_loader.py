import argparse
import torch

from medseg.data.msd import load_msd_dataset, fixed_split
from medseg.data.build_loader import build_loaders

"""
01_check_loader.py
验证的是:基本通路正确
数据读取没炸+transform没炸+loader没炸+batch结构符合训练要求
torch.isnan(t).any().item()
.any()的意思是只要有一个nan就返回tensor(True)
这个的.item()是取值，因为.any()返回的是tensor,所以需要.item()取值

t.is_floating_point()
判断t是不是浮点数类型

torch.isinf(t)
判断t是不是无穷大
Pytorc大结构
1.Tensor操作:
torch.sum,torch.mean,torch.std,torch.max,torch.min,torch.argmax,torch.argmin
2.数学函数
.exp,.log,.sqrt,.sin,.cos,.abs,.pow

3.逻辑函数
.isnan,.isinf,.isfinite,.all,.any,.where

4.tensor创建
.zeros,.ones,.rand,.randn,.arange,.linspace

5.shape操作
.reshape,.view,.permute,.transpose,.unsqueeze,.cat,.stack

6.神经网络
torch.nn.Conv2d,.Conv3d,.BatchNorm2d,.BatchNorm3d,.ReLU,.Sigmoid,.Softmax
torch.nn.functional

vars(args)把对象转换为字典
iter()的作用是创建迭代器
for batch in train_loader:
本质就是,1.创建iterator,2.不断next()迭代
StopIteration当迭代器遍历完所有元素后，会抛出StopIteration异常


"""



def check_tensor_basic(name, t):
    print(f"\n[{name}]")
    print("  shape :", tuple(t.shape))
    print("  dtype :", t.dtype)
    print("  device:", t.device)

    if not torch.is_tensor(t):
        print("  not a torch tensor!")
        return

    has_nan = torch.isnan(t).any().item() if t.is_floating_point() else False
    has_inf = torch.isinf(t).any().item() if t.is_floating_point() else False
    print("  has_nan:", has_nan)
    print("  has_inf:", has_inf)

    if t.numel() == 0:
        print("  empty tensor!")
        return

    if t.is_floating_point():
        print("  min   :", float(t.min().item()))
        print("  max   :", float(t.max().item()))
        print("  mean  :", float(t.mean().item()))
        print("  std   :", float(t.std().item()))
    else:
        print("  min   :", int(t.min().item()))
        print("  max   :", int(t.max().item()))


def check_label_unique(name, y, max_show=20):
    """
    检查label到底有那些类别值(0,1,2,..)打印出来
    torch.unique(y)返回tensor中所有不同的值
    
    """
    print(f"\n[{name} unique labels]")
    uniq = torch.unique(y)
    uniq_list = uniq.cpu().tolist()
    print("  unique:", uniq_list[:max_show])
    if len(uniq_list) > max_show:
        print(f"  ... total {len(uniq_list)} unique values")


def check_image_label_pair(x, y, pair_name="pair"):
    print(f"\n[{pair_name} image-label consistency]")

    if x.ndim != y.ndim:
        print(f"  ndim mismatch: image.ndim={x.ndim}, label.ndim={y.ndim}")
        return

    print(f"  image shape: {tuple(x.shape)}")
    print(f"  label shape: {tuple(y.shape)}")

    if x.shape[0] != y.shape[0]:
        print("  batch size mismatch!")
    else:
        print("  batch size matched")

    if x.ndim >= 5 and y.ndim >= 5:
        if x.shape[2:] == y.shape[2:]:
            print("  spatial shape matched")
        else:
            print("  spatial shape mismatch!")

    elif x.ndim >= 4 and y.ndim >= 4:
        if x.shape[1:] == y.shape[1:]:
            print("  non-batch dims matched")
        else:
            print("  non-batch dims mismatch!")


def check_batch_dict(batch, batch_name="batch"):
    print(f"\n========== check {batch_name} ==========")

    if not isinstance(batch, dict):
        print("batch is not dict!")
        print(type(batch))
        return

    print("keys:", list(batch.keys()))

    if "image" not in batch:
        print("丢失 key: image")
        return
    if "label" not in batch:
        print("丢失 key: label")
        return

    x = batch["image"]
    y = batch["label"]

    if not torch.is_tensor(x):
        print("image is not tensor!")
        print(type(x))
        return
    if not torch.is_tensor(y):
        print("label is not tensor!")
        print(type(y))
        return

    check_tensor_basic(f"{batch_name}/image", x)
    check_tensor_basic(f"{batch_name}/label", y)
    check_label_unique(f"{batch_name}/label", y)
    check_image_label_pair(x, y, pair_name=batch_name)

    if x.ndim not in (4, 5):
        print(f"warning: image ndim={x.ndim}, usually 3D med image batch expects 5D [B,C,D,H,W]")
    if y.ndim not in (4, 5):
        print(f"warning: label ndim={y.ndim}, usually 3D seg label batch expects 5D [B,C,D,H,W] or similar")

    if x.ndim == 5:
        print(f"image seems like [B,C,D,H,W] = {tuple(x.shape)}")
    if y.ndim == 5:
        print(f"label seems like [B,C,D,H,W] = {tuple(y.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--patch", type=int, nargs=3, default=[96, 96, 96])
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--cache_rate", type=float, default=0.2)
    ap.add_argument("--train_check_batches", type=int, default=3)
    args = ap.parse_args()

    print("========== args ==========")
    print(vars(args))

    print("\n========== load dataset ==========")
    train_items, test_items = load_msd_dataset(args.data_root)
    print("num train_items:", len(train_items))
    print("num test_items :", len(test_items))

    if len(train_items) == 0:
        raise RuntimeError("train_items is empty!")

    print("\n========== fixed split ==========")
    tr, va = fixed_split(train_items, val_ratio=args.val_ratio)
    print("num train split:", len(tr))
    print("num val split  :", len(va))

    if len(tr) == 0:
        raise RuntimeError("train split is empty!")
    if len(va) == 0:
        raise RuntimeError("val split is empty!")

    print("\n========== build loaders ==========")
    train_loader, val_loader = build_loaders(
        tr,
        va,
        patch_size=tuple(args.patch),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
    )

    print("train_loader built:", train_loader is not None)
    print("val_loader built  :", val_loader is not None)

    print("\n========== iterate train loader ==========")
    
    
    train_iter = iter(train_loader)
    for i in range(args.train_check_batches):
        try:
            batch = next(train_iter)
            check_batch_dict(batch, batch_name=f"train_batch_{i}")
        except StopIteration:
            print(f"train loader ended early at batch {i}")
            break
        except Exception as e:
            print(f"error while reading train batch {i}: {repr(e)}")
            raise

    print("\n========== iterate val loader ==========")
    try:
        vbatch = next(iter(val_loader))
        check_batch_dict(vbatch, batch_name="val_batch_0")
    except StopIteration:
        print("val loader is empty!")
        raise
    except Exception as e:
        print(f"error while reading val batch: {repr(e)}")
        raise

    print("\n========== cuda ==========")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        print("current device   :", torch.cuda.current_device())
        print("device name      :", torch.cuda.get_device_name(torch.cuda.current_device()))

    print("\n✅ loader basic check finished.")


if __name__ == "__main__":
    main()