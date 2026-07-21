#!/usr/bin/env python3
"""
将 data36 标签转换为纯四点格式，并删除 N(熄灭)/P(紫色) 类别。

旧格式: cls cx cy w h x1 y1 x2 y2 x3 y3 x4 y4 -1 -1  (15列)
新格式: cls x1 y1 x2 y2 x3 y3 x4 y4                  (9列)

类别处理:
    0-8   B(蓝色)  保留
    9-17  R(红色)  保留
    18-26 N(熄灭)  删除
    27-35 P(紫色)  删除
"""

import os
import glob
import argparse


def convert_file(path):
    """转换单个标签文件，返回 (保留行数, 删除行数)。"""
    kept, dropped = [], 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            if cls >= 18:  # N / P 类别删除
                dropped += 1
                continue
            if len(parts) >= 13:  # 旧格式: cls cx cy w h + 8个关键点 [+ -1 -1]
                kpts = parts[5:13]
            elif len(parts) == 9:  # 已是四点格式
                kpts = parts[1:9]
            else:
                print(f"警告: 跳过无效行 ({len(parts)} 列) in {path}")
                continue
            kept.append(f"{cls} {' '.join(kpts)}\n")
    with open(path, "w") as f:
        f.writelines(kept)
    return len(kept), dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data36", help="数据集根目录")
    args = parser.parse_args()

    for split in ["train", "test"]:
        label_dir = os.path.join(args.data, split, "labels")
        if not os.path.isdir(label_dir):
            print(f"目录不存在: {label_dir}")
            continue
        files = glob.glob(os.path.join(label_dir, "*.txt"))
        total_kept = total_dropped = 0
        for fp in files:
            k, d = convert_file(fp)
            total_kept += k
            total_dropped += d
        print(f"{split}: {len(files)} 个文件, 保留 {total_kept} 个目标, 删除 {total_dropped} 个 N/P 目标")

    # 删除旧缓存，避免使用过期标签
    for cache in glob.glob(os.path.join(args.data, "*", "*.cache")):
        os.remove(cache)
        print(f"已删除缓存: {cache}")


if __name__ == "__main__":
    main()
