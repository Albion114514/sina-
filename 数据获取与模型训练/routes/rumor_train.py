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
Stdout:
  - "Accuracy: <float>"     (for auto_pipeline parsing)
"""

import os
import sys
import json
import time
import math
import random
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, classification_report
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
    def __init__(
        self,
        content_df: pd.DataFrame,
        comment_df: pd.DataFrame,
        label_map: Dict[str, int],
        max_len: int = 128,
        max_comments: int = 10,
        vocab: Dict[str, int] = None,
        build_vocab: bool = False,
    ):
        """
        content_df columns: [id, keyword, text, reserved, timestamp]
        comment_df columns: [comment_id, content_id, name, text, timestamp]
        label_map: {id -> 0/1}
        """
        self.content_df = content_df
        self.comment_df = comment_df
        self.label_map = label_map
        self.max_len = max_len
        self.max_comments = max_comments
        self.vocab = vocab if vocab is not None else {"<PAD>": 0, "<UNK>": 1}

        if build_vocab:
            self._build_vocab()

        # 只保留在 label_map 中出现过的样本
        self.content_df = self.content_df[self.content_df[0].astype(str).isin(self.label_map.keys())].reset_index(drop=True)

    def _build_vocab(self):
        idx = max(self.vocab.values()) + 1 if self.vocab else 0
        if "<PAD>" not in self.vocab:
            self.vocab["<PAD>"] = idx; idx += 1
        if "<UNK>" not in self.vocab:
            self.vocab["<UNK>"] = idx; idx += 1
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


def main():
    args = parse_args()
    set_seed(args.seed)

    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
              "cuda" if args.device == "cuda" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CSVs
    content_df = pd.read_csv(args.content_path, header=False)  # [id, keyword, text, reserved, timestamp]
    comment_df = pd.read_csv(args.comment_path, header=False)  # [comment_id, content_id, name, text, timestamp]
    labels_df = pd.read_csv(args.labels_path, header=False)    # [id, label]

    # Build label map
    label_map = {str(r[0]): int(r[1]) for _, r in labels_df.iterrows()}

    # Train/Val split over content ids that have labels
    ids = content_df[0].astype(str).tolist()
    ids = [i for i in ids if i in label_map]
    train_ids, val_ids = train_test_split(ids, test_size=args.val_split, random_state=args.seed, stratify=[label_map[i] for i in ids])

    train_content = content_df[content_df[0].astype(str).isin(train_ids)].reset_index(drop=True)
    val_content   = content_df[content_df[0].astype(str).isin(val_ids)].reset_index(drop=True)

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

    # Model
    model = RumorDetector(
        vocab_size=len(train_ds.vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        pad_idx=train_ds.vocab["<PAD>"],
    ).to(device)

    # Train
    best_acc, history = train(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, patience=3
    )

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

    # Also echo artifact paths for debugging
    print(f"Saved: {best_model_path}")
    print(f"Saved: {vocab_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {pred_csv_path}")


if __name__ == "__main__":
    main()
