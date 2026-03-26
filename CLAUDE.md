# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 3D medical image segmentation framework built on PyTorch and MONAI, targeting liver and heart CT segmentation. Supports both online (on-the-fly) and offline (preprocessed `.pt` files) training pipelines.

## Common Commands

### Training
```bash
# Online mode (from .nii.gz files)
python scripts/train.py --task liver --model unet3d --epochs 200

# Offline mode (from preprocessed .pt files)
python scripts/train.py --task liver --model unet3d --epochs 200 --offline

# Key arguments
--task          # Task name: "liver" or "heart" (defined in medseg/tasks.py)
--model         # Model: unet3d, dynunet, unetr, swinunetr, attention_unet, segresnet
--patch_size    # Input patch size (default: 96 96 96)
--batch_size    # Training batch size
--overlap       # Sliding window overlap for validation (default: 0.5)
--loss          # Loss function: dicece, dicefocal, tversky, focaltversky
--binary        # Train with merged liver+tumor as single foreground
--amp           # Enable automatic mixed precision
--output_dir    # Where to save checkpoints and logs
```

### Evaluation
```bash
python scripts/eval.py --task liver --model unet3d --ckpt path/to/best.pt
```

### Tests
```bash
python -m pytest tests/test_forward.py
# Tests forward pass for all 6 model architectures with dummy [1,1,96,96,96] input
```

### Preprocessing Tools
```bash
python tools/preprocess_offline.py   # Convert .nii.gz → .pt (run before offline training)
python tools/calc_patch.py           # Compute optimal patch size from dataset
python tools/calc_ratios_nnunet.py   # Calculate foreground/background sampling ratios
python tools/calc_spacing.py         # Analyze voxel spacing statistics
python tools/calc_window.py          # Calculate CT intensity window bounds
```

### Data Debugging
```bash
python scripts/check/01_check_loader.py      # Inspect DataLoader batches
python scripts/check/check_speed.py          # Benchmark data loading speed
python tools/validate_pt_files.py            # Validate preprocessed .pt files
```

## Architecture

### Training Pipeline (`scripts/train.py` → `medseg/`)

```
train.py
  ├─ get_task()              # medseg/tasks.py: data_root + num_classes per task
  ├─ load_data()             # Online: load_msd_dataset() | Offline: load_pt_paths()
  ├─ split_three_ways()      # 80/10/10 train/val/test split
  ├─ build_loaders_auto()    # medseg/data/build_loader.py: creates DataLoaders
  ├─ build_model()           # medseg/models/build_model.py: model factory
  └─ training loop
       ├─ get_stage_ratios() # Dynamic fg/bg ratios that change mid-training
       ├─ train_one_epoch_softmax() / train_one_epoch_sigmoid_binary()
       └─ validate_sliding_window()
```

### Data Modes

**Online** (`medseg/data/transforms.py`): Full MONAI pipeline — load NIfTI → orient RAS → resample to 0.88mm isotropic → intensity normalize (CT window [-13.7, 188.3]) → crop foreground → random crop by label → augmentation.

**Offline** (`medseg/data/dataset_offline.py` + `medseg/data/transforms_offline.py`): Loads preprocessed `.pt` tensors; only applies random augmentations at runtime. Use `tools/preprocess_offline.py` to generate `.pt` files first.

### Models (`medseg/models/build_model.py`)

All models wrap MONAI implementations. Factory call: `build_model(name, in_channels=1, out_channels=num_classes, img_size=patch_size)`.

- `unet3d` — Classic 3D U-Net, channels (32,64,128,256,320), strides (2,2,2,2)
- `dynunet` — nnUNet-style dynamic U-Net (recommended for best performance)
- `swinunetr` — Swin Transformer U-Net
- `unetr` — ViT-based U-Net
- `attention_unet` — U-Net with attention gates
- `segresnet` — ResNet-based segmentation

### Loss Functions (`medseg/engine/train_eval.py`)

- `dicece` (default) — DiceCE loss
- `dicefocal` — Dice + Focal loss
- `tversky` / `focaltversky` — Tversky variants
- `medseg/engine/adaptive_loss.py` — LearnableWeightedLoss for binary liver/tumor with trainable class weights

### Task Configuration (`medseg/tasks.py`)

Add new tasks here with `data_root` and `num_classes`. Currently: `liver` (3 classes: background/liver/tumor) and `heart` (2 classes).

### Output Files (per training run)

Each run saves to `output_dir/`:
- `best.pt` / `last.pt` — Model checkpoints (full training state for resume)
- `log.csv` — Per-epoch: loss, dice scores, learning rate
- `config.json` — All training arguments
- `metrics.json` — Final best score, epoch, duration
- `report.txt` — Human-readable summary
- `cmd.txt` — Exact command used to launch training