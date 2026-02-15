#!/usr/bin/env python3
"""
将旧标签格式转换为新的4点格式
旧格式: cls cx cy w h x1 y1 x2 y2 x3 y3 x4 y4 [-1 -1] (14-15列)
新格式: cls x1 y1 x2 y2 x3 y3 x4 y4 (9列)
"""
import os
import glob
from pathlib import Path

def convert_label_file(input_path, output_path=None):
    """转换单个标签文件"""
    if output_path is None:
        output_path = input_path  # 原地覆盖
    
    new_lines = []
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 13:  # 旧格式：cls cx cy w h x1 y1 x2 y2 x3 y3 x4 y4 ...
                cls = parts[0]
                # 提取4个关键点坐标 (索引5-12)
                kpts = parts[5:13]
                new_line = f"{cls} {' '.join(kpts)}"
                new_lines.append(new_line)
            elif len(parts) == 9:  # 已经是新格式
                new_lines.append(line.strip())
            else:
                print(f"警告: 跳过无效行 ({len(parts)} 列): {line.strip()}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(new_lines))
        if new_lines:
            f.write('\n')

def convert_directory(label_dir):
    """转换目录下所有标签文件"""
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    print(f"找到 {len(label_files)} 个标签文件")
    
    converted = 0
    for label_file in label_files:
        try:
            convert_label_file(label_file)
            converted += 1
        except Exception as e:
            print(f"转换失败 {label_file}: {e}")
    
    print(f"成功转换 {converted} 个文件")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='转换标签格式')
    parser.add_argument('--data', type=str, default='./data36', help='数据集根目录')
    args = parser.parse_args()
    
    # 转换训练集和测试集
    for split in ['train', 'test']:
        label_dir = os.path.join(args.data, split, 'labels')
        if os.path.exists(label_dir):
            print(f"\n转换 {split} 标签...")
            convert_directory(label_dir)
        else:
            print(f"目录不存在: {label_dir}")
    
    print("\n转换完成!")
