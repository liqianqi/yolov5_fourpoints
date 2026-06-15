# YOLOv5 项目 Agent 指令

本项目为 YOLOv5 目标检测/关键点检测，使用自定义数据集 data36（36 类，4 关键点标注）。

---

## 训练 (Training)

### 数据集配置

- **数据配置**: `data/data36.yaml`
- **数据集路径**: `./data36`
- **类别数**: 36
- **标注格式**: 含 4 个关键点的目标检测

### 常用训练命令

```bash
# 单 GPU 训练（推荐从预训练权重开始）
python train.py --data data/data36.yaml --weights yolov5s.pt --img 640

# 多 GPU DDP 训练
python -m torch.distributed.run --nproc_per_node 4 train.py --data data/data36.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3
```

### 关键文件

- `train.py` - 训练入口
- `utils/dataloaders.py` - 数据加载与增强
- `utils/loss.py` - 损失计算
- `utils/augmentations.py` - 数据增强
- `models/yolo.py` - 模型定义

### 训练相关修改原则

- 修改数据加载时优先检查 `utils/dataloaders.py` 和 `utils/augmentations.py`
- 修改损失函数时检查 `utils/loss.py` 中的 `ComputeLoss`
- 修改模型结构时检查 `models/yolo.py` 和 `models/common.py`
- 超参数在 `data/hyps/` 下，可通过 `--hyp` 指定

---

## 代码风格

- 使用 Python 3，遵循 PEP 8
- 新增功能需与现有 YOLOv5 风格一致
- 回复使用简体中文
