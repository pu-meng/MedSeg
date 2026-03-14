# medseg/utils/experiment.py
"""
DiceLoss:医学分割
CrossEntropy:分类
DiceCE:医学分割
FocalLoss:小目标
Optimizer(优化器),才是真正更新模型参数的人
forward
↓
loss
↓
backward
↓
optimizer.step()
代码是
loss.backward()
optimizer.step()

这个experiment.py是"实验记录模块",专门记录实验信息
save_run_metadata(workdir,args)自动保存一次饰演的所有关键信息;
比如cmd.txt,config.json,run_info.txt

metadata=元数据;描述数据的数据
assert isinstance(workdir,(str,Path))
检查workdir是否是str或Path类型,如果不是,抛出异常
workdir一般指本次实验的输出目录
os =Operating System,操作系统接口库
os.listdir()列出目录下的所有文件
os.remove()删除文件

import sys
sys是系统信息模块
sys.argv是命令行参数
sys.exit()退出程序
sys.path是模块搜索路径
sys.version是Python版本
sys.platform是操作系统平台
sys.executable是Python解释器路径

如果我终端输入:python train.py --epochs 300
那么
sys.argv =
[
 "train.py",
 "--epochs",
 "300"
]


import json
json用来保存结构化数据
json.dump(data,file)把python对象写入JSON文件

datetime.now().isoformat()
这里的isoformat()是转换成标准格式:

torch.cuda.get_device_name(0)这个是获取GPU名字
比如:NVIDIA RTX 2080 Ti

如果args=parser.parse_args()
此时
args=Namespace(
    epochs=300,
    batch_size=16,
    )
类型是argparse.Namespace

用vars(args)
得到
{
"epoch": 300,
"batch_size": 16
}
JSON支持dict,list,int,float,str,bool



"""

import os
import sys
import json
import torch
from datetime import datetime
from pathlib import Path


def save_run_metadata(workdir, args):
    """
    保存:
    - 运行命令
    - 配置参数
    - 运行环境信息
    """
    assert isinstance(workdir, (str, Path)), (
        f"workdir should be a path, got {type(workdir)}"
    )
    os.makedirs(workdir, exist_ok=True)

    # 1️⃣ 保存命令行
    with open(os.path.join(workdir, "cmd.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # 2️⃣ 保存参数
    with open(os.path.join(workdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 3️⃣ 保存运行信息
    with open(os.path.join(workdir, "run_info.txt"), "w") as f:
        f.write(f"time: {datetime.now().isoformat()}\n")
        f.write(f"workdir: {workdir}\n")
        f.write(f"cuda_available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"gpu_name: {torch.cuda.get_device_name(0)}\n")
