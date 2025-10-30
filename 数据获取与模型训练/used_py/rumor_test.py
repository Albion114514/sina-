
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from collections import defaultdict
from datetime import datetime

from rumor_detect_model import RumorDetector

def read_formatted_csv(content_path, comments_path):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True, help="Path to weibo_data_format.csv")
    ap.add_argument("--comments", required=True, help="Path to comments_format.csv")
    ap.add_argument("--model_in", required=True, help="Path to load model .pth")
    ap.add_argument("--output", required=True, help="Where to write prediction_result.csv")
    args = ap.parse_args()

    rows = read_formatted_csv(args.content, args.comments)

    detector = RumorDetector()
    detector.load(args.model_in)

    # Write predictions CSV compatible with auto_pipeline expectations
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["content_id","prediction","confidence"])
        for cid, content, comments in rows:
            pred = detector.predict(content, comments)
            wr.writerow([cid, pred["prediction"], f"{pred['confidence']:.6f}"])

    # Print a friendly line; auto_pipeline doesn't require it in test mode,
    # but it's useful for logs.
    print("Predictions written:", args.output)

if __name__ == "__main__":
    main()
