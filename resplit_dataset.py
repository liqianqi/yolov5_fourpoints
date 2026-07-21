#!/usr/bin/env python3
"""
合并 train/test 后按类别分层重新划分数据集。

分层策略: 每张图按其包含的"全局最稀有类别"归入一个层，
背景图(无标签)单独一层，每层内按固定随机种子抽取 test_frac 到测试集。
保证每个类别在测试集中的占比与全集一致。
"""

import glob
import os
import random
import shutil
from collections import Counter

random.seed(0)
ROOT = "data36"
TEST_FRAC = 0.10
NC = 12

# 1. 收集所有样本 (图片路径, 标签路径, 类别集合)
samples = []
class_count = Counter()
for split in ["train", "test"]:
    for lb in sorted(glob.glob(f"{ROOT}/{split}/labels/*.txt")):
        im = lb.replace("/labels/", "/images/").replace(".txt", ".jpg")
        assert os.path.isfile(im), f"缺少图片: {im}"
        with open(lb) as f:
            classes = [int(float(l.split()[0])) for l in f.read().strip().splitlines() if l.strip()]
        class_count.update(classes)
        samples.append((im, lb, classes))

print(f"合并后共 {len(samples)} 张图, {sum(class_count.values())} 个目标")

# 2. 按最稀有类别分层
def stratum(classes):
    if not classes:
        return -1  # 背景图
    return min(classes, key=lambda c: class_count[c])

strata = {}
for s in samples:
    strata.setdefault(stratum(s[2]), []).append(s)

# 3. 每层内随机抽取 test
train_set, test_set = [], []
for key, items in sorted(strata.items()):
    random.shuffle(items)
    n_test = max(1, round(len(items) * TEST_FRAC)) if len(items) >= 5 else 0
    test_set += items[:n_test]
    train_set += items[n_test:]
    name = "背景" if key == -1 else f"类{key}"
    print(f"  层 {name:<4}: 共 {len(items):>5}, test {n_test}")

print(f"新划分: train {len(train_set)}, test {len(test_set)}")

# 4. 移动到临时目录再替换（避免覆盖冲突）
for split in ["train_new", "test_new"]:
    for sub in ["images", "labels"]:
        os.makedirs(f"{ROOT}/{split}/{sub}", exist_ok=True)

def move(items, split):
    for im, lb, _ in items:
        shutil.move(im, f"{ROOT}/{split}/images/{os.path.basename(im)}")
        shutil.move(lb, f"{ROOT}/{split}/labels/{os.path.basename(lb)}")

move(train_set, "train_new")
move(test_set, "test_new")

# 5. 替换旧目录
for split in ["train", "test"]:
    shutil.rmtree(f"{ROOT}/{split}")
    os.rename(f"{ROOT}/{split}_new", f"{ROOT}/{split}")

# 6. 清缓存
for cache in glob.glob(f"{ROOT}/*/*.cache") + glob.glob(f"{ROOT}/*.cache"):
    os.remove(cache)

# 7. 验证新分布
print("\n===== 新分布验证 =====")
for split in ["train", "test"]:
    cnt = Counter()
    files = glob.glob(f"{ROOT}/{split}/labels/*.txt")
    for lb in files:
        with open(lb) as f:
            cnt.update(int(float(l.split()[0])) for l in f.read().strip().splitlines() if l.strip())
    total = sum(cnt.values())
    dist = " ".join(f"{c}:{cnt.get(c, 0)}" for c in range(NC))
    print(f"{split}: {len(files)} 图 {total} 目标 | {dist}")
