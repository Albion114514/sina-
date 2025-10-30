#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_labels.py - 调试标签匹配问题
"""

import csv
import sys


def debug_labels(content_path, labels_path):
    """调试标签匹配问题"""
    print("🔍 调试标签匹配...")

    # 读取微博数据
    weibo_ids = set()
    with open(content_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if row:
                weibo_id = row[0].strip()
                weibo_ids.add(weibo_id)
                if i < 5:  # 打印前5个
                    print(f"微博CSV 行 {i + 1}: ID='{weibo_id}'")

    print(f"📊 微博CSV中的ID数量: {len(weibo_ids)}")

    # 读取标签数据
    label_ids = set()
    with open(labels_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) >= 2:
                label_id = row[0].strip()
                label_value = row[1]
                label_ids.add(label_id)
                if i < 5:  # 打印前5个
                    print(f"标签CSV 行 {i + 1}: ID='{label_id}', 标签='{label_value}'")

    print(f"📊 标签CSV中的ID数量: {len(label_ids)}")

    # 检查匹配
    matched = weibo_ids.intersection(label_ids)
    missing_in_labels = weibo_ids - label_ids
    extra_in_labels = label_ids - weibo_ids

    print(f"✅ 匹配的ID数量: {len(matched)}")
    print(f"❌ 微博中有但标签中缺失的ID数量: {len(missing_in_labels)}")
    print(f"⚠️  标签中有但微博中不存在的ID数量: {len(extra_in_labels)}")

    if missing_in_labels:
        print("前5个缺失的ID:")
        for id in list(missing_in_labels)[:5]:
            print(f"  '{id}'")

    if extra_in_labels:
        print("前5个多余的ID:")
        for id in list(extra_in_labels)[:5]:
            print(f"  '{id}'")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python debug_labels.py <微博CSV路径> <标签CSV路径>")
        sys.exit(1)

    debug_labels(sys.argv[1], sys.argv[2])