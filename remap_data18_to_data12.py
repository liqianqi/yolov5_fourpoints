#!/usr/bin/env python3
"""
将 18 类四点标签缩减为 12 类：删除 O(前哨站)/Bs(基地)/Bb(基地大装甲)，
只保留 G(哨兵) 和 1-5 号。

旧编号 (18类): cls = color * 9 + digit, color∈{0:B,1:R}, digit∈{0:G,1..5,6:O,7:Bs,8:Bb}
新编号 (12类): cls = color * 6 + digit, digit∈{0:G,1..5}

映射: B 0-5 → 0-5, R 9-14 → 6-11, 其余(6,7,8,15,16,17)删除
"""

import os
import glob
import argparse


def remap_file(path):
    kept, dropped = [], 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            color, digit = divmod(cls, 9)
            if digit >= 6:  # O / Bs / Bb 删除
                dropped += 1
                continue
            new_cls = color * 6 + digit
            kept.append(f"{new_cls} {' '.join(parts[1:9])}\n")
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
            k, d = remap_file(fp)
            total_kept += k
            total_dropped += d
        print(f"{split}: {len(files)} 个文件, 保留 {total_kept} 个目标, 删除 {total_dropped} 个 O/Bs/Bb 目标")

    for cache in glob.glob(os.path.join(args.data, "*", "*.cache")):
        os.remove(cache)
        print(f"已删除缓存: {cache}")


if __name__ == "__main__":
    main()
