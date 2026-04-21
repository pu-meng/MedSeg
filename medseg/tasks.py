# medseg/tasks.py
"""
任务注册表,,
让所有德数据集不需要修改train.py和eval.py

第一步:入口和配置

medseg_project/medseg/tasks.py    任务定义

第二步:数据管道

medseg_project/medseg/data/transforms.py    Online 预处理变换

medseg_project/medseg/data/dataset_offline.py+medseg_project/medseg/data/transforms_offline.py  Offline路径

第三步:模型

按需看具体模型文件

第四步:训练引擎
medseg_project/medseg/engine/adaptive_loss.py — Loss 设计 

整个是核心文件
scripts/train.py                                                                                            
medseg/tasks.py 
medseg/data/build_loader.py + transforms.py + dataset_offline.py                                            
medseg/engine/train_eval.py + adaptive_loss.py   

"""

TASKS = {
    # MSD Task02_Heart
    "heart": {
        "data_root": "/home/pumengyu/Task02_Heart",
        "num_classes": 2,  # 0=bg, 1=heart
        "class_names": ["bg", "heart"],
    },
    # MSD Task03_Liver (Liver + Tumor)
    "liver": {
        "data_root": "/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver",
        "num_classes": 3,  # 0=bg, 1=liver, 2=tumor
        "class_names": ["bg", "liver", "tumor"],
    },
}


def get_task(name: str) -> dict:
    name = (name or "").lower()
    if name not in TASKS:
        raise KeyError(f"未知的任务: {name}. 可以接受的输入: {sorted(TASKS.keys())}")
    return TASKS[name]
