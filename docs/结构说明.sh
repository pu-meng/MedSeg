segmentation/
│
├── medseg_project/              ← 项目根目录（永远在这里运行命令）
│
│   ├── medseg/                  ← 核心 Python 包
│   │   ├── data/                ← 数据读取 + transforms + loader
│   │   ├── models/              ← 网络结构
│   │   ├── engine/              ← 训练 / 验证 / 推理逻辑
│   │   ├── utils/               ← logger / ckpt / seed 等
│   │   └── __init__.py
│   │
│   ├── scripts/                 ← Python 可执行入口
│   │   ├── train.py
│   │   ├── eval.py
│   │   └── shell/               ← 你喜欢的 .sh 管理区
│   │       ├── train_heart.sh
│   │       ├── train_liver.sh
│   │       ├── debug.sh
│   │       └── ablation_patch.sh
│   │
│   ├── runs/                    ← 所有实验输出（永远不放代码）
│   │   ├── heart_unet3d/
│   │   ├── heart_unetr/
│   │   └── ...
│   │
│   ├── docs/                    ← 结构说明 / 实验记录（可选）
│   │
│   ├── requirements.txt
│   └── README.md
│
├── Task02_Heart/                ← 原始数据（永远不动）
└── legacy_ct/                   ← 旧项目