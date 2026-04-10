先运行calc_spacing.py，计算spacing，得到spacing.txt
然后运行preprocess.py，得到对应的数据集
然后运行calc_patch.py和calc_window.py，得到对应的patch和window

python -m tools.validate_pt_files \
  --pt_dir  /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Liver_0.88mm_pre \
  --src_dir /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm \
  --verbose   # 可选，打印每个文件的详细信息
  python -m tools.validate_pt_files \
     --pt_dir  /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Liver_0.88mm_pre \
     --src_dir /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver_0.88mm \
     --delete_src