# medseg/tasks.py
"""
任务注册表,,
让所有德数据集不需要修改train.py和eval.py
"""

TASKS = {
    # MSD Task02_Heart
    "heart": {
        "data_root": "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task02_Heart",
        "num_classes": 2,  # 0=bg, 1=heart
        "class_names": ["bg", "heart"],
    },
    # MSD Task03_Liver (Liver + Tumor)
    "liver": {
        "data_root": "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver",
        "num_classes": 3,  # 0=bg, 1=liver, 2=tumor
        "class_names": ["bg", "liver", "tumor"],
    },
}


def get_task(name: str) -> dict:
    name = (name or "").lower()
    if name not in TASKS:
        raise KeyError(f"未知的任务: {name}. 可以接受的输入: {sorted(TASKS.keys())}")
    return TASKS[name]
