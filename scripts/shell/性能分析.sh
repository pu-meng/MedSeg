watch -n 1 nvidia-smi

htop

nvidia-smi --query-gpu=utilization.gpu --format=csv

iostate -x 1


iostat -x 1
free -h
ps -ef | grep python
pkill -f scripts.train

| 环节    | 影响什么      | 工具         |
| ----- | --------- | ---------- |
| CPU   | 数据预处理     | htop       |
| 磁盘 IO | 读数据速度     | iostat 1  iotop   |
| 内存    | 是否触发 swap | htop  free -h      |
| GPU   | 模型计算      | nvidia-smi |

nvidia-smi dmon 
dmon(Device Monitoring) 这个是nvidi-smi的一个子命令,可以实时显示GPU的运行状态
训练慢nvidia-smi
htop
nvtop 是GPU版本的htop,比nvidia-smi dmon 更好用,更强大的


PID(Process ID) 进程编号


PCIe带宽,GEN 1@ 8x,
这里的GEN(generation)是PCIe的版本,GEN 1@ 8x,表示PCIe 1.0,8个lane
GEN 1是第一代,代数越高,单条车道的限速越高,页越快
8x指代车道数
RX(Receive) 接收带宽,代表数据从CPU/内存流入GPU的速度

TX(Transmit) 发送带宽,代表数据从GPU流入CPU/内存的速度
POW(Power)当前的功耗

服务器有两个物理PCIe插槽,每个插槽插了2个GPU,但是Gen1 和Gen3的状态既设计物理硬件,也涉及软件的动态管理
路是固定的,但是限速是动态的

pkill -u pumengyu -f "scripts.train" 关掉所有

ps aux | grep pumengyu | grep "scripts.train" | grep -v grep
# 没有输出 = 全关了
tmux new -s 名字

# 重新SSH登录服务器后
tmux ls                    # 看看会话还在不在
# liver_exp: 1 windows (created ...) ← 还在

tmux attach -t liver_exp   # 回去，看到完整输出，就像没断过

Ctrl+B  然后按  [       ← 进入"翻页模式"


Linux的内存策略不是等内存用完才用swap,吧长期不用的数据提前丢进swap
只有swap爆炸增长+机器卡顿才是问题
num_workers=cpu核心数的一半
num_workers=4~8较好
进程process,是一个程序实例,比如python train.py,chrome
线程thread,是进程的一个执行单元,
PyTorch训练的主线程,数据加载线程,OpenMP线程,MKL线程

内核线程kthr,系统内部的线程,比如IO调度,内存管理,网络管理
runing 当前正在占用CPU执行的线程数量
PID是递增分配的,除非机器重启不然PID不会重启


第一个查看 htop 
Tasks:188,698 thr,255 kthr;3 running
Load average: 1.70 0.63 0.22
分别是1分钟负载,5分钟负载,15分钟负载
Uptime 是重启时间
PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
  874 root      20   0  416820  87360  52688 S  0.0   0.1   0:00.00 bash
  PID process ID是进程编号
  PRI Priority 调度优先级,数值越小越优先
  NI Nice 值,数值越小越优先,人为调节优先值
  VRI 虚拟内存使用量,
  RES 实际物理内存使用
  SHR 共享内存使用
  S 进程状态.Running, S sleeping, T stopped, Z zombie(危险)
  %CPU CPU使用率
  %MEM 内存使用率
  TIME+ 进程累计运行时间
  COMMAND 进程名称

nvidia-smi
NVIDIA-SMI 565.57.01
Driver Version: 565.57.01
CUDA Version: 12.7
意思是:显卡驱动版本是565.57.01,CUDA版本是12.7
Fan 风扇转速百分比,Temp 温度,
Perf 性能状态,P0=满性能,P8=空闲状态
Pwr:Usage/Cap:功耗,显存使用量/显存容量
Memory-usage:130MiB / 11264MiB
GPU-Util:0% GPU使用率,这个是最重要的指标


iostat
看磁盘IO 是否为瓶颈
avg-cpu:  %user %nice %system %iowait %steal %idle
          0.79   0.00   0.22    0.02    0.00   98.97
%user: 用户程序占用CPU时间百分比
%system: 内核占用时间
%iowait: CPU在等待磁盘读写的时间比例
%idle: 空闲CPU时间百分比

Device:            tps    kB_read/s    kB_wrtn/s    kB_dscd/s    kB_read    kB_wrtn    kB_dscd