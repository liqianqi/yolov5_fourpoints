#!/usr/bin/env python3
"""
将 4关键点标注格式 转换为 YOLO 边界框格式.

原格式: class kp1_x kp1_y kp2_x kp2_y kp3_x kp3_y kp4_x kp4_y
新格式: class x_center y_center width height
"""

import os
from pathlib import Path


def keypoints_to_bbox(kp1_x, kp1_y, kp2_x, kp2_y, kp3_x, kp3_y, kp4_x, kp4_y):
    """将4个关键点转换为边界框 (x_center, y_center, width, height)."""
    # 找到所有关键点的最小外接矩形
    x_coords = [kp1_x, kp2_x, kp3_x, kp4_x]
    y_coords = [kp1_y, kp2_y, kp3_y, kp4_y]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # 计算中心点和宽高
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


def convert_label_file(input_path, output_path):
    """转换单个标签文件."""
    with open(input_path) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 9:  # 4关键点格式: class + 8个坐标
            cls = int(float(parts[0]))
            kp1_x, kp1_y = float(parts[1]), float(parts[2])
            kp2_x, kp2_y = float(parts[3]), float(parts[4])
            kp3_x, kp3_y = float(parts[5]), float(parts[6])
            kp4_x, kp4_y = float(parts[7]), float(parts[8])

            x_center, y_center, width, height = keypoints_to_bbox(
                kp1_x, kp1_y, kp2_x, kp2_y, kp3_x, kp3_y, kp4_x, kp4_y
            )

            # 确保值在 [0, 1] 范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0.001, min(1, width))
            height = max(0.001, min(1, height))

            new_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        elif len(parts) == 5:  # 已经是标准格式
            new_lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(new_lines)


def convert_dataset(input_dir, output_dir):
    """转换整个数据集."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录结构
    for split in ["train", "test"]:
        for subdir in ["images", "labels"]:
            (output_path / split / subdir).mkdir(parents=True, exist_ok=True)

    # 处理训练集和测试集
    for split in ["train", "test"]:
        labels_dir = input_path / split / "labels"
        images_dir = input_path / split / "images"

        if labels_dir.exists():
            print(f"转换 {split} 标签...")
            label_files = list(labels_dir.glob("*.txt"))
            for i, label_file in enumerate(label_files):
                convert_label_file(label_file, output_path / split / "labels" / label_file.name)
                if (i + 1) % 100 == 0:
                    print(f"  已处理 {i + 1}/{len(label_files)} 个文件")
            print(f"  完成: {len(label_files)} 个标签文件")

        # 复制/链接图片
        if images_dir.exists():
            print(f"链接 {split} 图片...")
            for img_file in images_dir.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    dst = output_path / split / "images" / img_file.name
                    if not dst.exists():
                        os.symlink(img_file.absolute(), dst)
            print("  完成")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="转换关键点标注为边界框格式")
    parser.add_argument("--input", default="data36", help="输入数据集路径")
    parser.add_argument("--output", default="data36_bbox", help="输出数据集路径")
    args = parser.parse_args()

    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print()

    convert_dataset(args.input, args.output)

    print()
    print("=" * 50)
    print("转换完成!")
    print(f"新数据集保存在: {args.output}")
    print()
    print("下一步:")
    print("1. 创建新的 data.yaml 配置文件指向新数据集")
    print("2. 重新训练模型")
