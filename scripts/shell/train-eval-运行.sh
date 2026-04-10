python -m scripts.01_check_loader \
  --data_root "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task02_Heart" \
  --batch_size 2 --num_workers 4




python -c "from medseg.data.msd import load_msd_dataset, fixed_split; print('import msd OK')"

python -W default -c "import torch"


查找问题来源


python -m scripts.train \
  --data_root "../Task02_Heart" \
  --workdir "./runs/heart_unet3d" \
  --epochs 50 --batch_size 2 --amp


python -m scripts.train \
  --data_root "../Task02_Heart" \
  --workdir "./experiments/runs/heart_unet3d" \
  --epochs 50 --batch_size 2 --amp



python -m scripts.eval \
  --data_root "../Task02_Heart" \
  --ckpt "../experiments/heart_unet3d/best.pt" \
  --patch 96 96 96


python -m scripts.summarize_run --run_dir "../experiments/heart_unet3d" --tail 10

#训练
python -m scripts.train \
  --data_root "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task02_Heart" \
  --workdir   "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/experiments/heart_unet3d" \
  --model unet3d --epochs 200 --batch_size 2 --patch 96 96 96 --amp

nohup python -m scripts.train \
  --data_root "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task02_Heart" \
  --workdir   "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/experiments/heart_unet3d" \
  --model unet3d --epochs 200 --batch_size 2 --patch 96 96 96 --amp \
  > /home/pumengyu/First2TB/PuMengYu/CT/segmentation/experiments/heart_unet3d/log.txt 2>&1 &
  
训练

python -m scripts.train \
  --data_root "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task02_Heart" \
  --exp_root "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/experiments" \
  --exp_name "heart_unet3d" \
  --model unet3d --epochs 100 --batch_size 2 --patch 96 96 96 --amp


#验证

python -m scripts.eval \
  --ckpt "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/experiments/heart_unet3d/train/02-28-22/best.pt" \
  --exp_name heart_unet3d

python -m scripts.eval \
  --data_root "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task02_Heart" \
  --ckpt "/home/pumengyu/First2TB/PuMengYu/CT/segmentation/experiments/heart_unet3d/train/03-01-16/best.pt" \
  --exp_name heart_unet3d



运行少量训练
python scripts/train.py --task liver --exp_name liver_debug --epochs 5 --train_n 8 --val_n 4 --batch_size 1 --cache_rate 0.0 --amp
#3D数据集数据大，batch_size大了容易OOM，cuda out of memory
#cache_rate 越大训练越快，占用内存越多
# batch_size 决定GPU每次处理多少数据，cache_rate决定内存中缓存多少数据
# num_workers决定CPU几个进程准备数据
watch -n 1 nvidia-smi
训练

CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
 --task liver \
 --exp_name liver_unet3d_200 \
 --mode  unet3d \
 --data_root /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver \
 --num_classes 3 --epochs 30 \
 --batch_size 2 --lr 1e-4 --patch 96 96 96 \
 --val_ratio 0.2  \
 --num_workers 9 --cache_rate 0.1 --sw_batch_size 2 \
 --seed 0 --loss dicefocal --amp

CUDA_VISIBLE_DEVICES=1 python -m scripts.eval \
  --task liver \
  --exp_name liver_unet3d_200 \
  --data_root /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver \
  --num_classes 3 \
  --model unet3d \
  --ckpt /home/pumengyu/First2TB/PuMengYu/CT/segmentation/experiments/liver_unet3d_200/train/03-03-21-45-34/best.pt \
  --patch 96 96 96 \
  --sw_batch_size 2 \
  --val_ratio 0.2 \
  --num_workers 6 \
  --cache_rate 0.1 \
  --seed 0

CUDA_VISIBLE_DEVICES=0 python -m tools.diag.eval_tumor_only \
  --task liver \
  --data_root /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver \
  --ckpt /home/pumengyu/First2TB/PuMengYu/CT/segmentation/experiments/liver_unet3d_200/train/03-03-21-45-34/best.pt \
  --model unet3d \
  --patch 96 96 96 \
  --sw_batch_size 2 \
  --val_ratio 0.2 \
  --seed 0 \
  --cache_rate 0.0
#我们现在用unetr来训练以下，保存到liver_unetr文件夹下
CUDA_VISIBLE_DEVICES=1 python -m scripts.train \
 --task liver \
 --exp_name liver_unetr \
 --data_root /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver \
 --num_classes 3 --epochs 10 \
 --batch_size 2 --lr 1e-4 --patch 96 96 96 \
 --val_ratio 0.2  \
 --num_workers 2 --cache_rate 0.1 --sw_batch_size 2 \
 --seed 0 --loss dicefocal --amp --model unetr


验证:注意epochs ,exp_name,ckpt,

CUDA_VISIBLE_DEVICES=1 python -m scripts.eval \
  --task liver \
  --exp_name liver_unetr \
  --mode unetr \
  --data_root /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_1p5mm \
  --num_classes 3 \
  --ckpt /home/pumengyu/First2TB/PuMengYu/CT/segmentation/experiments/liver_unet3d_stageRatio_ftversky/train/03-04-16-28-26/best.pt \
  --patch 80 80 80 \
  --sw_batch_size 2 \
  --val_ratio 0.2 \
  --num_workers 4 \
  --cache_rate 0.1 \
  --seed 0


# ====== config ======
GPU=0
TASK=liver
EXP=liver_unet3d_200
MODE=unet3d
DATA=/home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver

EPOCHS=3
BATCH=2
LR=1e-4
PATCH="96 96 96"
VAL_RATIO=0.2
NUM_WORKERS=8
CACHE=0.1
SW_BATCH=2
SEED=0
LOSS=dicefocal
AMP=--amp
# =====================

CUDA_VISIBLE_DEVICES=$GPU python -m scripts.train \
  --task $TASK \
  --exp_name $EXP \
  --mode $MODE \
  --data_root $DATA \
  --num_classes 3 \
  --epochs $EPOCHS \
  --batch_size $BATCH \
  --lr $LR \
  --patch $PATCH \
  --val_ratio $VAL_RATIO \
  --num_workers $NUM_WORKERS \
  --cache_rate $CACHE \
  --sw_batch_size $SW_BATCH \
  --seed $SEED \
  --loss $LOSS \
  $AMP


3月4号修改之后


CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --task liver \
  --exp_name liver_0.88mm_v1 \
  --data_root /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm \
  --num_classes 3 \
  --epochs 40 \
  --batch_size 2 \
  --lr 1e-4 \
  --patch 96 144 144 \
  --val_ratio 0.2 \
  --num_workers 12 \
  --cache_rate 0.0 \
  --sw_batch_size 2 \
  --val_every 5\
  --seed 0 \
  --amp \
  --loss dicefocal


CUDA_VISIBLE_DEVICES=1 python -m scripts.eval \
  --task liver \
  --exp_name liver_offline \
  --data_root /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm \
  --num_classes 3 \
  --ckpt /home/pumengyu/First2TB/PuMengYu/CT/segmentation/experiments/liver_offline_test/train/03-05-23-48-57/best.pt \
  --patch 144 144 144 \
  --sw_batch_size 2 \
  --val_ratio 0.2 \
  --num_workers 8 \
  --cache_rate 0.0 \
  --seed 0


离线offline数据集
训练

CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --task liver \
  --exp_name liver_offline \
  --preprocessed_root /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Liver_0.88mm_pre \
  --num_classes 3 \
  --epochs 100 \
  --batch_size 2 \
  --patch 96 96 96 \
  --val_every 3 \
  --num_workers 1 \
  --sw_batch_size 3 \
  --seed 0 \
  --prefetch_factor 2 \
  --amp \
  --repeats 1 \
  --loss dicefocal

测试 cache_rate在离线模式下没有用，
修改num_samples,和这个best_score

CUDA_VISIBLE_DEVICES=1 python -m scripts.eval \
  --task liver \
  --exp_name liver_offline \
  --num_classes 3 \
  --ckpt /home/pumengyu/First2TB/PuMengYu/CT/segmentation/experiments/liver_offline/train/03-08-11-05-39/last.pt \
  --patch 96 96 96 \
  --sw_batch_size 3 \
  --val_ratio 0.2 \
  --overlap 0.5 \
  --num_workers 1 \
  --cache_rate 0.0 \
  --seed 0 \
  --preprocessed_root /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Liver_0.88mm_pre


ls /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm/imagesTr | sort > /tmp/img_ids.txt
ls /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm/labelsTr | sort > /tmp/lab_ids.txt
diff /tmp/img_ids.txt /tmp/lab_ids.txt



在120新的战场,在线模式

CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --task liver \
  --exp_name liver_online_v0 \
  --data_root /home/PuMengYu/Task03_Liver \
  --num_classes 3 \
  --epochs 200 \
  --batch_size 2 \
  --patch 144 144 144 \
  --val_every 3 \
  --num_workers 4 \
  --sw_batch_size 4 \
  --seed 0 \
  --amp \
  --loss dicefocal


CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --task liver \
  --exp_name liver_online \
  --data_root /home/PuMengYu/Task03_Liver \
  --exp_root /home/PuMengYu/experiments \
  --num_classes 3 \
  --epochs 200 \
  --batch_size 2 \
  --patch 144 144 144 \
  --val_every 3 \
  --num_workers 2 \
  --sw_batch_size 4 \
  --cache_rate 0.1 \
  --amp \
  --loss dicefocal



CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
    --task liver \
    --exp_name liver_online \
    --data_root /home/PuMengYu/Task03_Liver \
    --exp_root /home/PuMengYu/experiments \
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
    --resume /home/PuMengYu/experiments/liver_online/train/03-09-09-06-43/last.pt \
2>&1 | tee train.log