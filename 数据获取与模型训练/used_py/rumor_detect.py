#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rumor_detect.py — unified entry that dispatches to train/test components.

Usage examples:
  Train:
    python rumor_detect.py --mode train --content data/weibo_content.csv --comments data/weibo_comments.csv --label 1 --output output/train_preds.csv

  Test:
    python rumor_detect.py --mode test --content data/weibo_content.csv --comments data/weibo_comments.csv --output output/test_preds.csv
"""
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Rumor detection dispatcher (train/test)")
    ap.add_argument('--mode', required=True, choices=['train', 'test'], help='Select train or test')
    ap.add_argument('--content', required=True, help='Formatted content CSV (no header)')
    ap.add_argument('--comments', required=True, help='Formatted comments CSV (no header)')
    ap.add_argument('--output', required=True, help='Output CSV path')

    # Train-only flags
    ap.add_argument('--label', type=int, choices=[1, 2], help='[train] 1=Rumor, 2=Non-rumor')
    ap.add_argument('--epochs', type=int, help='[train] Number of epochs')
    ap.add_argument('--batch_size', type=int, help='[train] Batch size')
    ap.add_argument('--lr', type=float, help='[train] Learning rate')
    ap.add_argument('--model_out', help='[train] Path to save trained model (default: models/rumor_lstm.pt)')

    # Test-only flags
    ap.add_argument('--model_in', help='[test] Path to trained model (default: models/rumor_lstm.pt)')

    return ap.parse_args()

def main():
    args = parse_args()

    here = Path(__file__).resolve().parent
    py = sys.executable

    if args.mode == 'train':
        if args.label is None:
            print("Error: --label is required in train mode (1=Rumor, 2=Non-rumor)")
            sys.exit(2)

        cmd = [py, str(here / 'rumor_train.py'),
               '--content', args.content,
               '--comments', args.comments,
               '--label', str(args.label),
               '--output', args.output]

        if args.epochs is not None:
            cmd += ['--epochs', str(args.epochs)]
        if args.batch_size is not None:
            cmd += ['--batch_size', str(args.batch_size)]
        if args.lr is not None:
            cmd += ['--lr', str(args.lr)]
        if args.model_out:
            cmd += ['--model_out', args.model_out]

    else:  # test
        cmd = [py, str(here / 'rumor_test.py'),
               '--content', args.content,
               '--comments', args.comments,
               '--output', args.output]
        if args.model_in:
            cmd += ['--model_in', args.model_in]

    # Ensure output directory exists (robustness)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Delegate to the chosen component
    try:
        completed = subprocess.run(cmd, check=False)
        sys.exit(completed.returncode)
    except FileNotFoundError as e:
        print(f"Failed to execute: {' '.join(cmd)}")
        print(str(e))
        sys.exit(1)

if __name__ == '__main__':
    main()
