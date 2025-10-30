
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import random

from rumor_detect_model import RumorDetector

def read_formatted_csv(content_path, comments_path):
    # content: [content_id, keyword, content_text, reserved, timestamp]
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

    # comments: [comment_id, content_id, comment_name, comment_text, timestamp]
    groups = defaultdict(list)
    with open(comments_path, "r", encoding="utf-8") as f:
        rd = csv.reader(f)
        for row in rd:
            if not row:
                continue
            wid = row[1]
            text = row[3]
            ts = int(float(row[4])) if row[4] else 0
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
            groups[wid].append({"text": text, "time": dt, "name": row[2], "reply2": ""})

    data = []
    for cid, content in contents.items():
        comments = groups.get(cid, [])
        data.append({"content": content, "comments": comments})
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True, help="Path to weibo_data_format.csv")
    ap.add_argument("--comments", required=True, help="Path to comments_format.csv")
    ap.add_argument("--label", required=True, type=int, choices=[1,2], help="1=rumer, 2=non-rumor")
    ap.add_argument("--model_out", required=True, help="Path to save model .pth")
    ap.add_argument("--output", required=True, help="Where to write prediction_result.csv (train preview)")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    # Assemble dataset with a single label applied to all rows
    dataset = read_formatted_csv(args.content, args.comments)
    label_value = 1 if args.label == 1 else 0
    for item in dataset:
        item["label"] = label_value

    # Shuffle & small split for a quick sanity accuracy
    random.seed(42)
    random.shuffle(dataset)
    if len(dataset) > 4:
        split = max(1, int(len(dataset)*0.8))
    else:
        split = max(1, len(dataset)-1)
    train_data = dataset[:split]
    val_data = dataset[split:] if split < len(dataset) else dataset[:1]

    detector = RumorDetector()
    detector.train(train_data, val_data=val_data, epochs=args.epochs, batch_size=4, learning_rate=0.001)

    # Evaluate and print the required "Accuracy: x.xxxx" line for auto_pipeline parsing
    metrics = detector.evaluate(val_data)
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    # Save model
    detector.save(args.model_out)

    # Write a tiny preview prediction_result.csv so downstream steps have a file
    # Format: content_id,prediction,confidence
    import csv
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["content_id","prediction","confidence"])
        for idx, item in enumerate(val_data):
            pred = detector.predict(item["content"], item["comments"])
            wr.writerow([f"val_{idx}", pred["prediction"], f"{pred['confidence']:.6f}"])

if __name__ == "__main__":
    main()
