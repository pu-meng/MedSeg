python -m tools.preprocess_offline \
  --src  /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm \
  --dst /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm_preprocessed \
  --num_workers 4



python -m tools.check_pt_labels --pt_dir /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Liver_0.88mm_pre --n 5


  python -m tools.calc_ratios_nnunet --pt_dir /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Liver_0.88mm_pre


CUDA_VISIBLE_DEVICES=1 python -m tools.calc_sw_batch_size --patch 144 144 144 --gpu_mem_gb 11



python calc_sw_batch_size.py --patch 144 144 144  # 自动检测显存