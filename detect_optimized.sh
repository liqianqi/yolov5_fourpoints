#!/bin/bash
# 优化后的检测脚本 - 基于exp4模型
# 使用验证后的最佳NMS参数

WEIGHTS="runs/train/exp4/weights/best.pt"
CONF_THRES=0.4 # 置信度阈值
IOU_THRES=0.3  # NMS IOU阈值 (降低后mAP提升5.5%)
IMG_SIZE=640
MAX_DET=50 # 每张图最大检测数

# 检测单张图片
# python detect.py --weights $WEIGHTS --source YOUR_IMAGE.jpg --conf-thres $CONF_THRES --iou-thres $IOU_THRES --img $IMG_SIZE --max-det $MAX_DET

# 检测图片文件夹
# python detect.py --weights $WEIGHTS --source YOUR_FOLDER/ --conf-thres $CONF_THRES --iou-thres $IOU_THRES --img $IMG_SIZE --max-det $MAX_DET

# 检测视频
# python detect.py --weights $WEIGHTS --source YOUR_VIDEO.mp4 --conf-thres $CONF_THRES --iou-thres $IOU_THRES --img $IMG_SIZE --max-det $MAX_DET

# 实时摄像头检测
# python detect.py --weights $WEIGHTS --source 0 --conf-thres $CONF_THRES --iou-thres $IOU_THRES --img $IMG_SIZE --max-det $MAX_DET

echo "=== 优化参数配置 ==="
echo "模型权重: $WEIGHTS"
echo "置信度阈值: $CONF_THRES"
echo "NMS IOU阈值: $IOU_THRES"
echo "图像尺寸: $IMG_SIZE"
echo "最大检测数: $MAX_DET"
echo ""
echo "使用示例:"
echo "  python detect.py --weights $WEIGHTS --source 图片路径 --conf-thres $CONF_THRES --iou-thres $IOU_THRES --img $IMG_SIZE"
