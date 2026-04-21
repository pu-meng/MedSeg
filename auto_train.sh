#!/bin/bash

# ===== 可改区域 =====
MEM_THRESHOLD=2000      # 显存占用低于这个值(MiB)认为基本空闲
CHECK_INTERVAL=60       # 每隔多少秒检查一次
LOG_FILE=wait_gpu.log   # 等待过程日志

TRAIN_CMD='python -u -m scripts.train \
    --task liver \
    --exp_name liver_online \
    --data_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver \
    --exp_root /home/PuMengYu/MSD_LiverTumorSeg/experiments \
    --num_classes 3 \
    --epochs 200 \
    --batch_size 2 \
    --patch 144 144 144 \
    --val_every 3 \
    --num_workers 2 \
    --sw_batch_size 4 \
    --cache_rate 0.3 \
    --amp \
    --loss dicefocal \
    --resume /home/PuMengYu/MSD_LiverTumorSeg/experiments/liver_online/train/03-09-09-06-43/last.pt'
# ===================

echo "===== $(date '+%F %T') auto_train start =====" | tee -a "$LOG_FILE"

while true
do
    mem0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
    mem1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')

    util0=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
    util1=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')

    now=$(date '+%F %T')
    echo "[$now] GPU0: mem=${mem0}MiB util=${util0}% | GPU1: mem=${mem1}MiB util=${util1}%" | tee -a "$LOG_FILE"

    # 优先选空闲的 GPU0，否则 GPU1
    if [ -n "$mem0" ] && [ "$mem0" -lt "$MEM_THRESHOLD" ]; then
        GPU_ID=0
        break
    fi

    if [ -n "$mem1" ] && [ "$mem1" -lt "$MEM_THRESHOLD" ]; then
        GPU_ID=1
        break
    fi

    echo "[$now] No free GPU, sleep ${CHECK_INTERVAL}s..." | tee -a "$LOG_FILE"
    sleep "$CHECK_INTERVAL"
done

echo "[$(date '+%F %T')] GPU${GPU_ID} is free. Start training." | tee -a "$LOG_FILE"

RUN_LOG="train_gpu${GPU_ID}_$(date '+%m-%d-%H-%M-%S').log"

CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "$TRAIN_CMD" 2>&1 | tee "$RUN_LOG"

echo "===== $(date '+%F %T') training finished =====" | tee -a "$LOG_FILE"
