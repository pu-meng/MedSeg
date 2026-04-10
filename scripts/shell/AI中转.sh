【项目名称】
medseg_project
【当前阶段】

Stage1:3D UNet baseline 稳定训练 + 规范化 train/eval 输出结构


【核心摘要-永久版本】

- medseg_project
- 3D medical segmentation
- train/eval分离
- experiments/<exp_name>/{train,eval}/<timestamp>/
- 自动保存 config + cmd + summary
- UNet3D / UNETR baseline

长期目标：
- 3D Transformer
- 多模态
- 可发论文结构
- 追求高薪就业

【目录结构】
experiments/
  heart_unet3d/
    train/<timestamp>/
      best.pt
      last.pt
      log.csv
      config.yaml
      cmd.txt
    eval/<timestamp>/
      summary.txt
      summary.json

【train.py 关键逻辑】
- 默认 data_root: /home/.../Task02_Heart
- 默认 exp_root: /home/.../segmentation/experiments
- 输出路径: exp_root/exp_name/train/<timestamp>
- 保存:
    - best.pt(当前最优)
    - last.pt(每10 epoch)
    - log.csv
    - config.yaml
    - cmd.txt

【eval.py 关键逻辑】

- 默认输出: exp_root/exp_name/eval/<timestamp>
- 批量验证整个 val 集
- 保存:
    - summary.txt
    - summary.json
    - 推理时间
    - 平均 Dice

【当前模型】
- UNet3D (MONAI)
- patch=96x96x96
- batch_size=2
- AMP可选

【当前问题】


