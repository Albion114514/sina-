# -*- coding: utf-8 -*-
"""
rumor_train.py
Train a dual-channel BiLSTM + MultiheadAttention rumor detector on Weibo CSVs.

Inputs:
  - weibo_data_format.csv: [id, keyword, text, reserved, timestamp] (no header)
  - comments_format.csv:   [comment_id, content_id, name, text, timestamp] (no header)
  - labels.csv:            [id, label] (no header, label in {0,1})

Outputs (under --output-dir):
  - best_model.pth
  - vocab.json
  - metrics.json
  - prediction_result.csv   (validation predictions: id,pred)

Stdout (for pipeline parsing):
  - "Accuracy: <float>"
"""

import os
import sys
import json
import time
import math
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Dataset & Collate
# ---------------------------
class RumorTrainDataset(Dataset):
    """
    content_df columns: [0:id, 1:keyword, 2:text, 3:reserved, 4:timestamp]
    comment_df columns: [0:comment_id, 1:content_id, 2:name, 3:text, 4:timestamp]
    """
    def __init__(
        self,
        content_df: pd.DataFrame,
        comment_df: pd.DataFrame,
        label_map: Dict[str, int],
        max_len: int = 128,
        max_comments: int = 10,
        vocab: Optional[Dict[str, int]] = None,
        build_vocab: bool = False,
    ):
        self.content_df = content_df
        self.comment_df = comment_df
        self.label_map = label_map
        self.max_len = max_len
        self.max_comments = max_comments
        self.vocab = vocab if vocab is not None else {"<PAD>": 0, "<UNK>": 1}

        # 只保留在 label_map 中出现过的样本
        self.content_df = self.content_df[self.content_df[0].astype(str).isin(self.label_map.keys())].reset_index(drop=True)

        if build_vocab:
            self._build_vocab()

    def _build_vocab(self):
        # 从已有词表继续累计
        idx = max(self.vocab.values()) + 1 if self.vocab else 0
        if "<PAD>" not in self.vocab:
            self.vocab["<PAD>"] = idx; idx += 1
        if "<UNK>" not in self.vocab:
            self.vocab["<UNK>"] = idx; idx += 1
        # 基于正文与评论构建词表
        for text in list(self.content_df[2].astype(str).values) + list(self.comment_df[3].astype(str).values):
            for tok in jieba.lcut(text):
                if tok not in self.vocab:
                    self.vocab[tok] = idx
                    idx += 1

    def _text2ids(self, text: str) -> torch.Tensor:
        toks = jieba.lcut(str(text))
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in toks][: self.max_len]
        if len(ids) == 0:
            ids = [self.vocab["<UNK>"]]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.content_df)

    def __getitem__(self, idx: int):
        row = self.content_df.iloc[idx]
        content_id = str(row[0])
        content_text = str(row[2])

        content_tensor = self._text2ids(content_text)

        # gather comments by content_id
        cmts = self.comment_df[self.comment_df[1].astype(str) == content_id][3].astype(str).tolist()
        cmt_tensors = [self._text2ids(t) for t in cmts[: self.max_comments]]

        label = int(self.label_map[content_id])
        return content_id, content_tensor, cmt_tensors, label


def dynamic_collate(batch):
    """
    batch: list of tuples (content_id, content_tensor, [cmt_tensors], label)
    Returns:
        content_ids: list[str]
        contents: LongTensor [B, S]
        comments: LongTensor [B, C, S]
        labels:   LongTensor [B]
    """
    content_ids, contents_ts, cmts_list, labels = zip(*batch)

    # 计算该 batch 的统一序列长度 S
    max_seq = 1
    for t in contents_ts:
        max_seq = max(max_seq, t.size(0))
    for cmts in cmts_list:
        for t in cmts:
            max_seq = max(max_seq, t.size(0))

    # 计算该 batch 的统一评论条数 C（取该 batch 的最大）
    max_cmts = max(1, max(len(cmts) for cmts in cmts_list))

    # pad contents -> [B, S]
    contents_pad = []
    for t in contents_ts:
        pad_len = max_seq - t.size(0)
        if pad_len > 0:
            t = torch.cat([t, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        contents_pad.append(t)
    contents = torch.stack(contents_pad, dim=0)

    # pad comments -> [B, C, S]
    comments_pad = []
    for cmts in cmts_list:
        # pad number of comments
        if len(cmts) < max_cmts:
            cmts = cmts + [torch.zeros(1, dtype=torch.long)] * (max_cmts - len(cmts))
        else:
            cmts = cmts[:max_cmts]

        # pad each comment to max_seq
        cmt_rows = []
        for t in cmts:
            pad_len = max_seq - t.size(0)
            if pad_len > 0:
                t = torch.cat([t, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            cmt_rows.append(t)
        comments_pad.append(torch.stack(cmt_rows, dim=0))

    comments = torch.stack(comments_pad, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(content_ids), contents, comments, labels


# ---------------------------
# Model
# ---------------------------
class RumorDetector(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_heads: int = 8,
        num_classes: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.content_lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.comment_lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_branch(self, x: torch.Tensor, lstm: nn.LSTM):
        # x: [B, S]
        emb = self.embedding(x)                  # [B, S, E]
        out, _ = lstm(emb)                       # [B, S, 2H]
        attn_out, _ = self.attn(out, out, out)   # [B, S, 2H]
        feat = attn_out.mean(dim=1)              # [B, 2H]
        return feat

    def forward(self, contents: torch.Tensor, comments: torch.Tensor):
        # contents: [B, S]
        # comments: [B, C, S]
        b, c, s = comments.size()
        cmt_flat = comments.view(b * c, s)
        content_feat = self.encode_branch(contents, self.content_lstm)        # [B, 2H]
        comment_feat = self.encode_branch(cmt_flat, self.comment_lstm)        # [B*C, 2H]
        comment_feat = comment_feat.view(b, c, -1).mean(dim=1)                # [B, 2H]
        fusion = torch.cat([content_feat, comment_feat], dim=-1)              # [B, 4H]
        logits = self.fc(fusion)                                              # [B, 2]
        return logits


# ---------------------------
# Train / Eval
# ---------------------------
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 3,
) -> Tuple[float, Dict]:
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for _, contents, comments, labels in train_loader:
            contents, comments, labels = contents.to(device), comments.to(device), labels.to(device)
            optim.zero_grad()
            logits = model(contents, comments)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        # Evaluate
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        history["train_loss"].append(total_loss / max(1, len(train_loader)))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"[Epoch {ep}] train_loss={history['train_loss'][-1]:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {ep}")
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc, history


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_labels, all_preds = [], []
    total_loss = 0.0

    for _, contents, comments, labels in loader:
        contents, comments, labels = contents.to(device), comments.to(device), labels.to(device)
        logits = model(contents, comments)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro") if all_labels else 0.0
    return avg_loss, acc, f1


@torch.no_grad()
def dump_validation_predictions(
    model: nn.Module,
    loader: DataLoader,
    out_csv: str,
):
    model.eval()
    out_rows = []
    for content_ids, contents, comments, _ in loader:
        logits = model(contents.to(next(model.parameters()).device),
                       comments.to(next(model.parameters()).device))
        preds = logits.argmax(dim=-1).cpu().tolist()
        for cid, p in zip(content_ids, preds):
            out_rows.append([cid, p])
    df = pd.DataFrame(out_rows)
    df.to_csv(out_csv, header=False, index=False)


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content-path", default="weibo_data_format.csv", type=str,
                    help="Path to weibo_data_format.csv")
    ap.add_argument("--comment-path", default="comments_format.csv", type=str,
                    help="Path to comments_format.csv")
    ap.add_argument("--labels-path", required=True, type=str,
                    help="Path to labels csv: [id,label], no header")
    ap.add_argument("--output-dir", default="model_output", type=str)
    ap.add_argument("--pretrained-path", type=str, default=None,
                    help="Optional path to an existing best_model.pth to continue training")
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-comments", type=int, default=10)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return ap.parse_args()


def _sync_latest(best_model_path: str, vocab_path: str, metrics_path: str, output_dir: str):
    """
    将最新成果同步到 models/latest/ 目录，便于测试阶段直接复用。
    """
    try:
        latest_dir = os.path.join(os.path.dirname(output_dir.rstrip("/\\")), "..", "models", "latest")
        latest_dir = os.path.abspath(latest_dir)
        os.makedirs(latest_dir, exist_ok=True)
        import shutil
        shutil.copyfile(best_model_path, os.path.join(latest_dir, "best_model.pth"))
        shutil.copyfile(vocab_path,        os.path.join(latest_dir, "vocab.json"))
        shutil.copyfile(metrics_path,      os.path.join(latest_dir, "metrics.json"))
        with open(os.path.join(latest_dir, "source_run.txt"), "w", encoding="utf-8") as f:
            f.write(f"source_run_dir={os.path.abspath(output_dir)}\n")
        print(f"🔄 Synced latest model & vocab into: {latest_dir}")
    except Exception as e:
        print(f"[WARN] Failed to sync latest artifacts: {e}")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
              "cuda" if args.device == "cuda" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 开始训练谣言检测模型...")
    print(f"📁 内容文件: {args.content_path}")
    print(f"📁 评论文件: {args.comment_path}")
    print(f"📁 标签文件: {args.labels_path}")
    print(f"📊 训练参数: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")

    # Load CSVs
    content_df = pd.read_csv(args.content_path, header=None)  # [id, keyword, text, reserved, timestamp]
    comment_df = pd.read_csv(args.comment_path, header=None)  # [comment_id, content_id, name, text, timestamp]
    labels_df = pd.read_csv(args.labels_path, header=None)  # [id, label]

    print(f"📊 读取到 {len(content_df)} 条微博数据，{len(comment_df)} 条评论")
    # Build label map
    label_map = {str(r[0]): int(r[1]) for _, r in labels_df.iterrows()}

    # 可视化检查（可选）
    preview = list(label_map.items())[:3]
    for i, (cid, lab) in enumerate(preview, 1):
        print(f"  标签 {i}: ID={cid}, label={lab}")
    print(f"✅ 成功加载 {len(label_map)} 个标签")

    # Train/Val split over content ids that have labels
    ids = content_df[0].astype(str).tolist()
    ids = [i for i in ids if i in label_map]
    print(f"✅ 加载数据: {len(ids)} 条微博, {len(label_map)} 个标签")
    if len(ids) == 0:
        print("[ERROR] 无可用的带标签样本，终止。")
        sys.exit(1)

    # 打印前三条 ID
    print("📝 前3条微博ID:")
    for i, cid in enumerate(ids[:3], 1):
        print(f"  微博 {i}: ID={cid}")

    # 统计匹配情况
    matched = len(ids)
    print("🎯 标签匹配结果:")
    print(f"  匹配成功: {matched} 条")
    print(f"  匹配失败: {len(label_map) - matched if len(label_map) >= matched else 0} 条")
    print(f"  前3个匹配的ID: {ids[:3]}")

    # split
    if len(set(label_map[i] for i in ids)) < 2:
        # 单类数据时 stratify 会报错，退化为随机切分
        train_ids, val_ids = train_test_split(ids, test_size=args.val_split, random_state=args.seed)
    else:
        train_ids, val_ids = train_test_split(ids, test_size=args.val_split, random_state=args.seed,
                                              stratify=[label_map[i] for i in ids])

    train_content = content_df[content_df[0].astype(str).isin(train_ids)].reset_index(drop=True)
    val_content   = content_df[content_df[0].astype(str).isin(val_ids)].reset_index(drop=True)

    print(f"🎯 可用于训练的数据: {len(ids)} 条")
    print(f"📚 训练集: {len(train_content)} 条, 验证集: {len(val_content)} 条")

    # Dataset (build vocab on training set)
    train_ds = RumorTrainDataset(
        train_content, comment_df, label_map,
        max_len=args.max_len, max_comments=args.max_comments,
        vocab=None, build_vocab=True
    )
    val_ds = RumorTrainDataset(
        val_content, comment_df, label_map,
        max_len=args.max_len, max_comments=args.max_comments,
        vocab=train_ds.vocab, build_vocab=False
    )

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=dynamic_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=dynamic_collate)

    # Model（注意：用“位置参数”传 vocab_size 更稳妥）
    model = RumorDetector(
        len(train_ds.vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        pad_idx=train_ds.vocab["<PAD>"],
    ).to(device)

    # 续训：若提供了旧模型路径，则加载其权重继续训练
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        try:
            print(f"🔁 Loading pretrained model from: {args.pretrained_path}")
            state = torch.load(args.pretrained_path, map_location=device)
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"[WARN] Failed to load pretrained weights: {e}")

    # Train
    print("🔧 开始训练模型...")
    best_acc, history = train(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, patience=3
    )
    print(f"📈 验证集准确率: {best_acc:.4f}")

    # Save best model and vocab/metrics
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    torch.save(model.state_dict(), best_model_path)

    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(train_ds.vocab, f, ensure_ascii=False)

    metrics = {
        "best_val_accuracy": best_acc,
        "history": history,
        "vocab_size": len(train_ds.vocab),
        "params": vars(args),
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Dump validation predictions as prediction_result.csv for pipeline consumption
    pred_csv_path = os.path.join(args.output_dir, "prediction_result.csv")
    dump_validation_predictions(model, val_loader, pred_csv_path)

    # IMPORTANT: print Accuracy for auto_pipeline parsing
    print(f"Accuracy: {best_acc:.6f}")

    # Echo artifact paths
    print(f"💾 模型已保存: {best_model_path}")
    print(f"💾 词表已保存: {vocab_path}")
    print(f"💾 指标已保存: {metrics_path}")
    print(f"📄 验证集预测: {pred_csv_path}")

    # 同步至 models/latest/ 便于测试阶段直接使用最新模型
    _sync_latest(best_model_path, vocab_path, metrics_path, args.output_dir)

    print("🎉 训练完成!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        raise
