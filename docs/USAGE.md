# 四点装甲板检测使用说明

本仓库基于 YOLOv5，改造为 **4 关键点四边形回归 + 颜色/贴纸双分支分类**，面向 RoboMaster 装甲板检测，并提供 TensorRT C++ 部署示例。

---

## 1. 项目概览

| 模块 | 说明 |
|------|------|
| 检测目标 | 装甲板四边形（4 个角点） |
| 分类结构 | 颜色分支（B/R）× 贴纸分支（G,1,2,3,4,5）→ 联合 12 类 |
| 轻量模型 | `models/yolov5-shufflenetv2.yaml`（推荐部署） |
| 数据配置 | `data/data12.yaml` |
| 超参数 | `data/hyp.kpt.yaml` |
| C++ 推理 | `trt_cpp/` |

联合类别编号：`cls = color * ndigit + digit`（当前 `ndigit=6`，`nc=12`）。

---

## 2. 环境准备

```bash
cd yolov5_fourpoints

# 创建虚拟环境（非 conda）
python3 -m venv venv
source venv/bin/activate

# 安装依赖（GPU 建议安装与驱动匹配的 torch）
pip install -U pip
pip install -r requirements.txt
# 若 CUDA 驱动为 12.8，可改用：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

C++ TensorRT 额外依赖：

- CUDA Toolkit（含 `nvcc`、`libcudart`）
- OpenCV 开发包（`libopencv-dev`）
- TensorRT 运行库（可用 pip：`pip install tensorrt-cu12`）
- TensorRT 头文件（见第 7 节）

---

## 3. 数据集

### 3.1 目录结构

```text
data36/
├── train/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

标签格式（每行 9 列，归一化坐标）：

```text
cls x1 y1 x2 y2 x3 y3 x4 y4
```

### 3.2 从旧格式转换

若原始标签是 `cls cx cy w h x1..y4 -1 -1`（15 列），且含 N/P 颜色类：

```bash
python convert_data36_to_fourpoints.py --data data36
```

会删除 `cls >= 18` 的 N/P 样本，并转为纯四点格式。

### 3.3 缩减到 12 类（去掉 O/Bs/Bb）

```bash
python remap_data18_to_data12.py --data data36
```

映射规则：

- 保留：G、1、2、3、4、5
- 删除：O（前哨站）、Bs（基地）、Bb（基地大装甲）
- 新编号：`cls = color * 6 + digit`，范围 `0~11`

### 3.4 分层重划分 train/test（可选）

```bash
python resplit_dataset.py
```

按“图内最稀有类别”分层抽样，默认约 10% 进测试集。

数据配置见 `data/data12.yaml`（`path: ./data36`）。

> `data36/` 已加入 `.gitignore`，不会推送到远程。

---

## 4. 训练

推荐命令（ShuffleNetV2，从头训练）：

```bash
source venv/bin/activate

python train.py \
  --cfg models/yolov5-shufflenetv2.yaml \
  --weights '' \
  --data data/data12.yaml \
  --hyp data/hyp.kpt.yaml \
  --img 640 \
  --batch-size 128 \
  --epochs 300 \
  --device 0 \
  --name shufflev2-fixed
```

也可用标准 backbone：

```bash
python train.py \
  --data data/data12.yaml \
  --hyp data/hyp.kpt.yaml \
  --weights yolov5s.pt \
  --img 640 \
  --batch-size 16 \
  --epochs 100 \
  --device 0 \
  --name data12
```

权重输出：`runs/train/<name>/weights/best.pt`

### 关键点解码说明（重要）

当前使用 **anchor 尺度归一 + 线性输出**（无 sigmoid）：

```text
kpt_pixel = raw * anchor_wh + cell_corner
```

旧版 `sigmoid*2-0.5` 只能表示 ±1 个 grid，会导致大目标角点饱和；请勿回退。

---

## 5. 验证与推理

```bash
# 验证
python val.py \
  --weights runs/train/shufflev2-fixed/weights/best.pt \
  --data data/data12.yaml \
  --img 640

# 推理
python detect.py \
  --weights runs/train/shufflev2-fixed/weights/best.pt \
  --source data36/test/images \
  --conf-thres 0.4 \
  --iou-thres 0.3 \
  --img 640
```

参考指标（ShuffleNetV2，300 epoch，测试集分层重划后）：

- mAP@0.5 ≈ 0.936
- mAP@0.5:0.95 ≈ 0.745
- Precision / Recall ≈ 0.90 / 0.92

---

## 6. 导出 ONNX（FP16）

```bash
python export.py \
  --weights runs/train/shufflev2-fixed/weights/best.pt \
  --include onnx \
  --img 640 \
  --device 0 \
  --half
```

得到：`runs/train/shufflev2-fixed/weights/best.onnx`（FP16）。  
可重命名为 `best-fp16.onnx` 便于区分。

ONNX 约定：

- 输入：`images`，形状 `[1, 3, 640, 640]`
- 输出：`output0`，形状 `[1, 25200, 21]`  
  （`8` 关键点 + `1` obj + `12` 类）

---

## 7. TensorRT C++ 推理

### 7.1 准备 TensorRT 头文件

```bash
cd trt_cpp
mkdir -p third_party
cd third_party
git clone --depth 1 --branch release/11.1 --filter=blob:none --sparse \
  https://github.com/NVIDIA/TensorRT.git trt_oss
cd trt_oss && git sparse-checkout set include
```

库文件默认使用 venv 中的 `tensorrt_libs`（见 `CMakeLists.txt`）。若系统已安装 TensorRT，可改链接路径。

### 7.2 编译

```bash
cd trt_cpp
cmake -B build -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
# 可选：IDE 索引
ln -sf build/compile_commands.json compile_commands.json
```

### 7.3 运行

```bash
./build/trt_detect \
  ../runs/train/shufflev2-fixed/weights/best-fp16.onnx \
  best-fp16.engine \
  ../data36/test/images \
  ./out_vis \
  0.4 0.3
```

参数顺序：

1. ONNX 路径（建议 FP16）
2. engine 缓存路径（不存在则自动构建）
3. 图片目录
4. 可视化输出目录
5. `conf`（默认 0.4）
6. `iou`（默认 0.3）

输出：

- `out_vis/`：画好四边形与类别的结果图
- 终端：预处理 / 推理 / 后处理耗时与 FPS

> `trt_cpp/build/`、`trt_cpp/out_vis/`、`*.engine` 已加入 `.gitignore`。

### 7.4 部署到 Jetson NX

- engine **不可跨设备复用**，请在 NX 上用同一份 ONNX 重新构建
- 头文件 / TensorRT 版本需与板端匹配
- 若增加类别，需同步改 `main.cpp` 中的 `NUM_CLS`、`NUM_COLS`、`CLASS_NAMES`

---

## 8. 增加新贴纸类别时改什么

例如加回前哨站 O、基地 Bs：

| 位置 | 修改 |
|------|------|
| `data/*.yaml` | `ndigit: 6→8`，`nc: 12→16`，补全 `names` |
| `models/yolov5-shufflenetv2.yaml` | `nc: 12→16`（`ncolor` 仍为 2） |
| 标签文件 | 按 `cls = color * ndigit + digit` 重新编号 |
| `trt_cpp/main.cpp` | `NUM_CLS`、`NUM_COLS(=9+nc)`、`CLASS_NAMES` |

Detect 头会自动计算 `ndigit = nc // ncolor`，改完后需**重新训练**。

---

## 9. 常用脚本一览

| 脚本 | 作用 |
|------|------|
| `convert_data36_to_fourpoints.py` | 15 列 → 9 列四点，删 N/P |
| `remap_data18_to_data12.py` | 18 类 → 12 类（去 O/Bs/Bb） |
| `resplit_dataset.py` | 分层重划分 train/test |
| `train.py` / `val.py` / `detect.py` | 训练 / 验证 / 推理 |
| `export.py` | 导出 ONNX 等格式 |
| `trt_cpp/main.cpp` | TensorRT C++ 推理 |

---

## 10. 相关文档

- `docs/四点回归模型修改说明.md`：四点改造细节
- `docs/NMS_CHANGES.md`：多边形 NMS / IoMin 说明
- `AGENTS.md`：Agent 协作约定
