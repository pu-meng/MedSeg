PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --task liver \
  --exp_name dynunet_v1_sgd \
  --model dynunet \
  --data_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver \
  --preprocessed_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt \
  --num_classes 3 \
  --epochs 300 \
  --batch_size 1 \
  --lr 0.01 \
  --patch 144 144 144 \
  --val_patch 96 96 96 \
  --sw_batch_size 1 \
  --val_every 6 \
  --num_workers 2 \
  --cache_rate 0.0 \
  --late_ratios 0.0 1.0 0.0 \
  --early_ratios 0.0 1.0 0.0 \
  --amp \
  --loss dicece \
  --overlap 0.5 \
  --prefetch_factor 4 \
  --repeats 3




two-stage 调整成0=背景,1=肝脏+肿瘤的二分类任务：
workdir = os.path.join(args.exp_root, args.exp_name, "train", timestamp)


# ============================================================
# eval — dynunet_liver_tumor_stage1 三分类，val集验证
# 结果保存在 experiments/dynunet_liver_tumor_stage1/eval/03-29-21-29-13/
# ============================================================

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/pumengyu:$PYTHONPATH python -m scripts.eval \
  --ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_tumor_stage1/train/03-29-21-29-13/best.pt \
  --preprocessed_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt \
  --model dynunet \
  --num_classes 3 \
  --patch 96 96 96 \
  --sw_batch_size 1 \
  --overlap 0.25 \
  --val_ratio 0.2 \
  --test_ratio 0.1 \
  --seed 0 \
  --split val \
  --min_tumor_size 100 \
  --save_vis


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --task liver \
  --exp_name dynunet_liver_only \
  --model dynunet \
  --data_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver \
  --preprocessed_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt \
  --num_classes 2 \
  --epochs 200 \
  --batch_size 1 \
  --lr 0.01 \
  --patch 144 144 144 \
  --val_patch 96 96 96 \
  --sw_batch_size 1 \
  --val_every 6 \
  --num_workers 2 \
  --cache_rate 0.0 \
  --early_ratios 0.0 1.0 \
  --late_ratios 0.0 1.0 \
  --merge_label12_to1 \
  --amp \
  --loss dicece \
  --overlap 0.5 \
  --prefetch_factor 4 \
  --repeats 3