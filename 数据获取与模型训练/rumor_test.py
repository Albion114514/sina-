#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rumor_test.py - 谣言检测模型测试脚本
用于加载训练好的模型并进行预测
"""

import os
import sys
import argparse
import csv
from collections import defaultdict
from datetime import datetime

# 导入模型
from rumor_detect_model import RumorDetector


def read_formatted_csv(content_path, comments_path):
    """读取格式化的CSV数据"""
    contents = {}
    with open(content_path, "r", encoding="utf-8") as f:
        rd = csv.reader(f)
        for row in rd:
            if not row:
                continue
            cid = row[0]
            text = row[2]
            ts = int(float(row[4])) if row[4] else 0
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
            contents[cid] = {"text": text, "time": dt}

    groups = defaultdict(list)
    with open(comments_path, "r", encoding="utf-8") as f:
        rd = csv.reader(f)
        for row in rd:
            if not row:
                continue
            wid = row[1]
            ts = int(float(row[4])) if row[4] else 0
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
            groups[wid].append({"text": row[3], "time": dt, "name": row[2], "reply2": ""})

    ordered = []
    for cid, content in contents.items():
        ordered.append((cid, content, groups.get(cid, [])))
    return ordered


def main():
    parser = argparse.ArgumentParser(description="测试谣言检测模型")
    parser.add_argument("--content-path", required=True, help="微博内容CSV文件路径")
    parser.add_argument("--comment-path", required=True, help="评论数据CSV文件路径")
    parser.add_argument("--model-path", required=True, help="训练好的模型路径")
    parser.add_argument("--output-path", required=True, help="预测结果输出路径")

    args = parser.parse_args()

    print("🧪 开始测试谣言检测模型...")
    print(f"📁 内容文件: {args.content_path}")
    print(f"📁 评论文件: {args.comment_path}")
    print(f"🤖 模型文件: {args.model_path}")

    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        return 1

    # 加载数据
    try:
        rows = read_formatted_csv(args.content_path, args.comment_path)
        print(f"✅ 加载数据: {len(rows)} 条微博")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return 1

    # 加载模型
    try:
        detector = RumorDetector()
        detector.load(args.model_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return 1

    # 进行预测
    try:
        with open(args.output_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["content_id", "prediction", "confidence"])

            total = len(rows)
            for i, (cid, content, comments) in enumerate(rows, 1):
                if i % 10 == 0:
                    print(f"📊 预测进度: {i}/{total}")

                pred = detector.predict(content, comments)
                wr.writerow([cid, pred["prediction"], f"{pred['confidence']:.6f}"])

        print(f"✅ 预测完成! 结果保存至: {args.output_path}")
        print(f"📄 总共预测: {total} 条微博")

        return 0

    except Exception as e:
        print(f"❌ 预测过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())