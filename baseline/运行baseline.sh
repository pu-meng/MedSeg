# 1. 自动规划（统计spacing/patch/窗口，约5分钟）
nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity

# 2. 训练（5折交叉验证，只训练第0折）
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 3 3d_fullres 0

# 3. 推理
nnUNetv2_predict \
    -i /path/to/test_images \
    -o /path/to/output \
    -d 3 \
    -c 3d_fullres \
    -f 0
```

---

## nnUNet自动帮你做的事
```
你之前手动做的              nnUNet自动做
─────────────────────────────────────────
统计spacing           →   自动统计，选中位数
计算patch大小          →   根据显存自动算
窗口归一化参数          →   自动统计0.5%~99.5%
ratios                →   自动按类别频率设置
LR scheduler          →   自动用PolyLR 1000epoch
数据增强               →   自动配置全套增强
```

---

## 预期结果
```
nnUNet在MSD Task03_Liver的结果（论文数据）：
  Liver Dice：0.963
  Tumor Dice：0.746

你用自己框架的合理目标：
  Liver Dice：0.92+
  Tumor Dice：0.45+


nnUNET的使用

第一步:设置环境变量
export nnUNet_raw="/home/pumengyu/First2TB/PuMengYu/CT/segmentation/nnunet/raw"
export nnUNet_preprocessed="/home/pumengyu/First2TB/PuMengYu/CT/segmentation/nnunet/preprocessed"
export nnUNet_results="/home/pumengyu/First2TB/PuMengYu/CT/segmentation/nnunet/results"

第二步：自动规划和预处理

nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity

20-30分钟 CPU运算
第三步：训练
 CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 0

40-50小时 GPU运算;
第四步：查看训练进度
# 实时看训练log
tail -f $nnUNet_results/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log*.txt
