export nnUNet_raw="/你的路径/nnUNet_raw"
export nnUNet_preprocessed="/你的路径/nnUNet_preprocessed"
export nnUNet_results="/你的路径/nnUNet_results"

# 自动规划预处理参数（spacing/patch/网络结构全自动）
nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity

# 开始训练（fold 0，3D fullres）
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 3 3d_fullres 0

# 推理
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 3 -c 3d_fullres -f 0


# 继续上次中断的训练
nnUNetv2_train 3 3d_fullres 0 --c

# 只跑 50 个 epoch（默认 1000，调试用）
nnUNetv2_train 3 3d_fullres 0 --num_epochs 50

# 在 plan_and_preprocess 阶段限制 patch size
nnUNetv2_plan_and_preprocess -d 3 --gpu_memory_target 10  # 单位 GB，你的 2080Ti 是 11GB

CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 3 3d_fullres 0