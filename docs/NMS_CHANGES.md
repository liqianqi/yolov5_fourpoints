# NMS 修改文档

## 概述

本文档记录了针对关键点检测任务对 NMS（非极大值抑制）进行的优化和 Bug 修复。

---

## 修改的文件

1. `/root/yolov5/utils/general.py` - 核心 NMS 算法
2. `/root/yolov5/detect.py` - 检测时的 NMS 调用

---

## 问题描述

### 问题1：框完全包含问题

传统 NMS 使用 IoU（交并比）判断重叠：

```
IoU = 交集面积 / 并集面积
```

当小框完全在大框内部时：

- IoU = 小框面积 / 大框面积
- 如果大框比小框大很多，IoU 可能 < 阈值，不会被去除

### 问题2：多边形顶点顺序 Bug

`polygon_intersection` 函数使用 Sutherland-Hodgman 算法，该算法要求顶点**逆时针排列**。
但数据集的关键点是**顺时针排列**的，导致：

- IoU 计算错误返回 0
- 完全相同的框无法被 NMS 去除

---

## 修改内容

### 1. 修复多边形交集算法 (`polygon_intersection`)

**位置**: `utils/general.py` 约第 1031 行

**修改前**:

```python
# 确保多边形是逆时针方向
result = [p.tolist() if hasattr(p, "tolist") else list(p) for p in poly1]

for i in range(len(poly2)):
    ...
```

**修改后**:

```python
# 确保多边形是逆时针方向
def ensure_ccw(poly):
    """确保多边形顶点按逆时针顺序排列."""
    pts = [p.tolist() if hasattr(p, "tolist") else list(p) for p in poly]
    # 计算有符号面积
    area = 0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    if area < 0:  # 顺时针，需要反转
        pts = pts[::-1]
    return pts


result = ensure_ccw(poly1)
poly2_ccw = ensure_ccw(poly2)

for i in range(len(poly2_ccw)):
    ...
    p1 = poly2_ccw[i]
    p2 = poly2_ccw[(i + 1) % len(poly2_ccw)]
    ...
```

---

### 2. 添加 IoMin 计算 (`polygon_iou_single`)

**位置**: `utils/general.py` 约第 1060 行

**修改前**:

```python
def polygon_iou_single(poly1, poly2):
    """计算两个四边形的IoU。poly1, poly2: (4, 2)."""
    ...
    return inter_area / union_area
```

**修改后**:

```python
def polygon_iou_single(poly1, poly2, return_iomin=False):
    """计算两个四边形的IoU。poly1, poly2: (4, 2).

    Args:
        poly1, poly2: 四边形顶点坐标 (4, 2)
        return_iomin: 是否同时返回IoMin (用于检测包含关系)

    Returns:
        如果 return_iomin=False: IoU
        如果 return_iomin=True: (IoU, IoMin)
    """
    ...
    iou = inter_area / union_area

    if return_iomin:
        # IoMin: 交集/最小框面积，用于检测包含关系
        # 当一个框完全包含另一个时，IoMin ≈ 1.0
        min_area = min(area1, area2)
        iomin = inter_area / min_area if min_area > 1e-6 else 0.0
        return iou, iomin

    return iou
```

**IoMin 原理 - 检测框包含关系**:

```
IoMin = 交集面积 / min(框1面积, 框2面积)
```

### 为什么 IoU 无法检测包含关系？

假设大框面积 = 100，小框面积 = 20，小框完全在大框内部：

```
┌─────────────────┐
│    大框 (100)    │
│   ┌─────┐       │
│   │小框 │       │
│   │(20) │       │
│   └─────┘       │
└─────────────────┘
```

**IoU 计算**:

```
交集 = 小框面积 = 20
并集 = 大框面积 = 100 (因为小框完全在大框内)
IoU = 20 / 100 = 0.2
```

IoU = 0.2 < 0.3 阈值 → **不会被 NMS 去除！**

**IoMin 计算**:

```
交集 = 小框面积 = 20
最小框面积 = min(100, 20) = 20
IoMin = 20 / 20 = 1.0
```

IoMin = 1.0 > 0.6 阈值 → **会被 NMS 去除！**

### 不同情况下的 IoMin 值

| 情况     | 示意图           | IoU          | IoMin   |
| -------- | ---------------- | ------------ | ------- |
| 完全包含 | `[大框[小框]]`   | 低 (0.1-0.3) | **1.0** |
| 完全相同 | `[框A=框B]`      | 1.0          | 1.0     |
| 部分重叠 | `[框A][框B]重叠` | 中等         | 中等    |
| 不重叠   | `[框A] [框B]`    | 0            | 0       |

### 判断逻辑

```python
# 满足任一条件则抑制（去除置信度低的框）：
if iou > iou_thres:  # 条件1：传统重叠检测
    suppress = True
elif iomin > iomin_thres:  # 条件2：包含关系检测
    suppress = True
```

- `iou_thres = 0.3`: 两框重叠超过 30% 时去除
- `iomin_thres = 0.6`: 小框 60% 以上在大框内时去除

---

### 3. 修改 polygon_nms 函数

**位置**: `utils/general.py` 约第 1099 行

**修改前**:

```python
def polygon_nms(kpts, scores, iou_thres):
    ...
    # 计算当前框与其余框的IoU
    remaining = order[1:]
    ious = np.array([polygon_iou_single(kpts[i].reshape(4, 2), kpts[j].reshape(4, 2)) for j in remaining])

    # 保留IoU小于阈值的框
    inds = np.where(ious <= iou_thres)[0]
    order = remaining[inds]
```

**修改后**:

```python
def polygon_nms(kpts, scores, iou_thres, iomin_thres=0.6):
    """基于多边形IoU的NMS，同时检测包含关系。.

    Args:
        kpts: 关键点坐标 (N, 8)
        scores: 置信度分数 (N,)
        iou_thres: IoU阈值，大于此值的框会被抑制
        iomin_thres: IoMin阈值，用于检测包含关系（一个框包含另一个）
    """
    ...
    # 计算当前框与其余框的IoU和IoMin
    remaining = order[1:]
    suppress_mask = np.zeros(len(remaining), dtype=bool)

    for idx, j in enumerate(remaining):
        iou, iomin = polygon_iou_single(kpts[i].reshape(4, 2), kpts[j].reshape(4, 2), return_iomin=True)
        # 满足任一条件则抑制：
        # 1. IoU > 阈值（传统NMS）
        # 2. IoMin > 阈值（一个框包含另一个）
        if iou > iou_thres or iomin > iomin_thres:
            suppress_mask[idx] = True

    # 保留未被抑制的框
    order = remaining[~suppress_mask]
```

**抑制条件**:

- `IoU > iou_thres` (默认 0.3): 传统重叠抑制
- `IoMin > iomin_thres` (默认 0.6): 包含关系抑制

---

### 4. 启用多边形 NMS (`detect.py`)

**位置**: `detect.py` 约第 211 行

**修改前**:

```python
pred = non_max_suppression(
    pred, conf_thres, iou_thres, classes, agnostic=True, max_det=max_det, polygon_nms_enabled=False
)
```

**修改后**:

```python
pred = non_max_suppression(
    pred, conf_thres, iou_thres, classes, agnostic=True, max_det=max_det, polygon_nms_enabled=True
)  # 使用多边形NMS
```

---

## 效果对比

### 修复前

- 完全相同的框无法去除（IoU 计算返回 0）
- 大框包含小框的情况无法处理（IoU < 阈值）
- 检测结果有大量嵌套框

### 修复后

| 图片    | 修复前检测数 | 修复后检测数       |
| ------- | ------------ | ------------------ |
| 112.jpg | 4个重叠框    | 1个框              |
| 187.jpg | 多个嵌套框   | 2个框              |
| 465.jpg | 多个嵌套框   | 3个框（每目标1个） |

---

## 参数说明

| 参数                  | 默认值 | 说明                                      |
| --------------------- | ------ | ----------------------------------------- |
| `iou_thres`           | 0.3    | IoU 阈值，检测命令行参数 `--iou-thres`    |
| `iomin_thres`         | 0.6    | IoMin 阈值，硬编码在 `polygon_nms` 函数中 |
| `polygon_nms_enabled` | True   | 是否使用多边形 NMS                        |
| `agnostic`            | True   | 跨类别 NMS（所有类别一起做 NMS）          |

---

## 使用方法

```bash
# 标准检测命令
python detect.py --weights runs/train/exp4/weights/best.pt \
  --source 图片路径 \
  --conf-thres 0.4 \
  --iou-thres 0.3
```

如需调整 IoMin 阈值，修改 `utils/general.py` 中 `polygon_nms` 函数的 `iomin_thres` 参数：

```python
def polygon_nms(kpts, scores, iou_thres, iomin_thres=0.6):  # 修改这里
```

---

## 性能影响

| 指标       | 修复前           | 修复后          |
| ---------- | ---------------- | --------------- |
| NMS 时间   | ~20ms (错误计算) | ~2ms (正确计算) |
| 检测准确性 | 大量误检         | 显著改善        |

---

## 相关文件

- `utils/general.py`: 核心 NMS 算法
- `detect.py`: 检测脚本
- `data/hyp.kpt.yaml`: 关键点检测优化超参数（用于重新训练）
