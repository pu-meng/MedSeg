tar -xvf Task02_Heart.tar
tar -tf Task02_Heart.tar > /dev/null

find Task02_Heart -name "._*" -delete

rm -rf src

mkdir -p medseg/{data,engine,models,utils} scripts runs
touch medseg/__init__.py


mkdir experiments
mv medseg_project/runs experiments/
/home/pumengyu/First2TB/PuMengYu/CT/segmentation

压缩

tar -cvzf medseg_code.tar.gz medseg_project \
  --exclude="*.pt"
  
解压缩 -x extract解压, -v verbose 显示过程,-f file文件名

tar -xvf Task03_Liver.tar
nproc

watch -n 1 nvidia-smi

python train.py > train.log 2>&1 &
把所有的输出都写进train.log文件中
tar -tf data.tar > /dev/null 只关心报 没报错   

command 2> /dev/null 屏蔽错误信息,
0是标准输入,1是标准输出,2是标准错误输出。

echo "hello world" > test.txt 生成简单文件

ls | wc -l
这里的ls命令会列出当前目录下的所有文件和文件夹,然后通过管道符（|）将输出传递给wc -l命令,wc -l命令会计算输入的行数,并将结果输出到标准输出。因此,这个命令的作用是计算当前目录下的文件和文件夹的数量。
|是管道符,用于将前一个命令的输出作为后一个命令的输入。
mkdir test || echo "创建失败"
这里的||表示如果前面的命令mkdir test执行失败（返回非零值）,则执行后面的echo "创建失败"命令。

echo $? 查看上一个命令的返回值
python train.py & 将命令放到后台执行,不占用当前终端


nohup python train.py > train.log 2>&1 &
nohup命令用于在后台运行命令,并且将标准输出和标准错误输出重定向到指定的文件中。在这个例子中,nohup命令将python train.py命令放到后台运行,并将标准输出和标准错误输出都重定向到train.log文件中。这样即使终端关闭,命令也会继续在后台运行,并且输出结果会保存在train.log文件中。
这样即使终端关闭,命令也会继续在后台运行,并且输出结果会保存在train.log文件中。

python train.py | tee train.log
tee是一边打印到屏幕,一边写入文件
python train.py | tee train.log

nohup python train.py > logs/train_$(date +%F).log 2>&1 &

cat是concatenate(拼接)
cat file1.txt file2.txt > all.txt
拼接文件
echo $(date) 等价于echo `date`当前的时间
date+%F 2023-10-11这里的date是当前的时间,%F是格式化日期的格式,表示年-月-日

python train.py &这里的&表示将命令放到后台执行,不占用当前终端
2>&1这里的&是将标准错误输出重定向到标准输出,这样标准输出和标准错误输出都会被重定向到train.log文件中。
2>1> train.log这里的2>表示将标准错误输出重定向到train.log文件中,1>表示将标准输出重定向到1文件中。

看文件夹大小
du -sh /home/pumengyu/First2TB/PuMengYu/CT/segmentation/Task03_Liver

python - <<'PY'
import monai
from monai.transforms import RandCropByLabelClassesd
print("monai version:", monai.__version__)
print("RandCropByLabelClassesd OK")
PY

ls Task03_Liver/imagesTr | wc -l
ls Task03_Liver_1p5mm/imagesTr | wc -l

ls Task03_Liver/labelsTr | wc -l
ls Task03_Liver_1p5mm/labelsTr | wc -l

简单检查:

tar -tzf Task03_Liver.tar.gz | head

du=disk usage(磁盘使用情况),--max-depth=1表示只显示当前目录下的文件和文件夹的大小,不显示子目录的大小。
-h 人类可读,sort -hr表示按照大小从大到小排序。
du -h --max-depth=1 | sort -hr


tmux new -s 名字

# 重新SSH登录服务器后
tmux ls                    # 看看会话还在不在
# liver_exp: 1 windows (created ...) ← 还在

tmux attach -t liver_exp   # 回去,看到完整输出,就像没断过
tmux kill-session -t pu 删除名字叫做pu的tmux会话
iostat -x -k -p 1 1
x是显示扩展统计信息,k是显示KB/s,p是显示分区,1是每隔1秒显示一次,1是显示1次。


tar -cvzf medseg_project.tar.gz \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  medseg_project/

tar -cvzf medseg_code.tar.gz medseg_project \
--exclude="__pycache__" \
--exclude="*.pt"
# 1. 安装包已经装完了没用了，直接删
rm /home/pumengyu/Anaconda3-2025.06-0-Linux-x86_64.sh

# 2. 看看还有啥大文件没显示完
du -sh /home/pumengyu/* --exclude=/home/pumengyu/First2TB 2>/dev/null | sort -rh


du -h --max-depth=3 / --exclude=/home/pumengyu/First2TB --exclude=/home/pumengyu/Second2TB 2>/dev/null | sort -rh | head -30


lsblk  看磁盘情况
df -h && echo "---" && lsblk

mkdir -p /media/8T/PuMengYu

ln -s /media/8T/PuMengYu /home/PuMengYu/8T


sudo swapoff -a
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h  # 验证，swap 应该变成 32G
