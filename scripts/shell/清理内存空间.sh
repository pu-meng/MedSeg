# 看First2TB下各目录大小
du -sh /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Liver_0.88mm_pre/* | sort -rh | head -5

# 看根分区下各目录大小
du -sh /home/pumengyu/* | sort -rh | head -20


第二步:清理__pycache__缓存
# 找到所有__pycache__并删除
find /home/pumengyu/First2TB -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 找到所有.pyc文件并删除
find /home/pumengyu/First2TB -name "*.pyc" -delete 2>/dev/null

echo "清理完成"


# conda下载缓存，通常几GB
conda clean --all -y

pip cache purge

