.
├── baseline
│   ├── convert_to_nnunet.py
│   └── 运行baseline.sh
├── docs
│   ├── 终端命令
│   │   └── scripts.sh
│   ├── 结构说明.sh
│   └── 论文.ipynb
├── medseg
│   ├── data
│   │   ├── build_loader.py
│   │   ├── dataset_offline.py
│   │   ├── __init__.py
│   │   ├── msd.ipynb
│   │   ├── msd.py
│   │   ├── __pycache__
│   │   │   ├── build_loader.cpython-310.pyc
│   │   │   ├── dataset_offline.cpython-310.pyc
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── msd.cpython-310.pyc
│   │   │   ├── transforms.cpython-310.pyc
│   │   │   └── transforms_offline.cpython-310.pyc
│   │   ├── transforms_offline.py
│   │   └── transforms.py
│   ├── engine
│   │   ├── __pycache__
│   │   │   └── train_eval.cpython-310.pyc
│   │   └── train_eval.py
│   ├── __init__.py
│   ├── models
│   │   ├── build_model.py
│   │   ├── __pycache__
│   │   │   ├── build_model.cpython-310.pyc
│   │   │   ├── unet3d.cpython-310.pyc
│   │   │   └── unetr.cpython-310.pyc
│   │   ├── unet3d.py
│   │   └── unetr.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   └── tasks.cpython-310.pyc
│   ├── tasks.py
│   └── utils
│       ├── ckpt.py
│       ├── experiment.py
│       ├── io_utils.py
│       ├── logger.py
│       ├── __pycache__
│       │   ├── ckpt.cpython-310.pyc
│       │   ├── io_utils.cpython-310.pyc
│       │   ├── logger.cpython-310.pyc
│       │   └── warnings.cpython-310.pyc
│       └── warnings.py
├── scripts
│   ├── 01_check_loader.py
│   ├── check
│   │   ├── debug_data.py
│   │   └── 检查.ipynb
│   ├── check_transforms_tumor.py
│   ├── eval.py
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── eval.cpython-310.pyc
│   │   ├── __init__.cpython-310.pyc
│   │   └── train.cpython-310.pyc
│   ├── shell
│   │   ├── AI中转.sh
│   │   ├── train-eval-运行.sh
│   │   ├── 性能分析.sh
│   │   ├── 清理内存空间.sh
│   │   └── 终端.sh
│   ├── summarize_run.py
│   └── train.py
└── tools
    ├── calc_patch.py
    ├── calc_ratios_nnunet.py
    ├── calc_ratios.py
    ├── calc_spacing.py
    ├── calc_sw_batch_size.py
    ├── calc_window.py
    ├── check_pt_labels.py
    ├── diag
    │   ├── debug_pred_tumor.py
    │   ├── eval_tumor_only.py
    │   ├── inspect_dataset.py
    │   └── tools_运行.sh
    ├── preprocess_offline.py
    ├── preprocess_resample.py
    ├── __pycache__
    │   └── validate_pt_files.cpython-310.pyc
    ├── tools-运行.sh
    ├── validate_pt_files.py
    └── 说明.sh

21 directories, 72 files