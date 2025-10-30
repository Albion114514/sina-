import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import jieba
import os
import pandas as pd
from datetime import datetime


# 数据集类（训练时需处理标签和批量数据）
class RumorDataset(Dataset):
    def __init__(self, contents, comments_list, labels, vocab, max_content_len=200, max_comment_len=50,
                 max_comments=100):
        self.contents = contents
        self.comments_list = comments_list
        self.labels = labels
        self.vocab = vocab
        self.max_content_len = max_content_len
        self.max_comment_len = max_comment_len
        self.max_comments = max_comments

    def __len__(self):
        return len(self.labels)

    def text_to_sequence(self, text):
        words = jieba.lcut(text)
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in words]

    def pad_sequence(self, sequence, max_len):
        if len(sequence) > max_len:
            return sequence[:max_len]
        return sequence + [self.vocab['<PAD>']] * (max_len - len(sequence))

    def __getitem__(self, idx):
        content_text = self.contents[idx]['text']
        content_seq = self.text_to_sequence(content_text)
        content_padded = self.pad_sequence(content_seq, self.max_content_len)

        comments = self.comments_list[idx][:self.max_comments]
        comments_processed = []
        for comment in comments:
            comment_seq = self.text_to_sequence(comment['text'])
            comment_padded = self.pad_sequence(comment_seq, self.max_comment_len)
            comments_processed.append(comment_padded)

        while len(comments_processed) < self.max_comments:
            comments_processed.append([self.vocab['<PAD>']] * self.max_comment_len)

        return {
            'content': torch.tensor(content_padded, dtype=torch.long),
            'comments': torch.tensor(comments_processed, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# 模型结构（训练和预测共享，需在两个组件中保持一致）
class RumorLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(RumorLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.content_lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                                    batch_first=True, dropout=dropout, bidirectional=True)
        self.comment_lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                                    batch_first=True, dropout=dropout, bidirectional=True)
        self.content_attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.comment_attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, content, comments):
        batch_size = content.size(0)
        num_comments = comments.size(1)

        # 处理内容文本
        content_embedded = self.embedding(content)
        content_output, _ = self.content_lstm(content_embedded)
        content_output = content_output.transpose(0, 1)
        content_attn, _ = self.content_attention(content_output, content_output, content_output)
        content_pooled = torch.mean(content_attn.transpose(0, 1), dim=1)

        # 处理评论
        comments_embedded = self.embedding(comments)
        comments_reshaped = comments_embedded.view(batch_size * num_comments, -1, comments_embedded.size(-1))
        comments_output, _ = self.comment_lstm(comments_reshaped)
        comments_output = comments_output.transpose(0, 1)
        comments_attn, _ = self.comment_attention(comments_output, comments_output, comments_output)
        comments_pooled = torch.mean(comments_attn.transpose(0, 1), dim=1)
        comments_aggregated = torch.mean(comments_pooled.view(batch_size, num_comments, -1), dim=1)

        # 融合特征并输出
        combined = torch.cat([content_pooled, comments_aggregated], dim=1)
        return self.fc(combined)


# 训练器类
class RumorTrainer:
    def __init__(self, embedding_dim=128, hidden_dim=64, n_layers=2, dropout=0.3):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.vocab = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_vocab(self, texts, min_freq=2):
        """构建词汇表（训练时专用）"""
        word_counter = Counter()
        for text in texts:
            word_counter.update(jieba.lcut(text))
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for word, count in word_counter.items():
            if count >= min_freq:
                vocab[word] = idx
                idx += 1
        self.vocab = vocab
        return vocab

    def preprocess_data(self, data):
        """预处理训练数据（含标签）"""
        contents, comments_list, labels, all_texts = [], [], [], []
        for item in data:
            contents.append(item['content'])
            comments_list.append(item['comments'])
            labels.append(item['label'])
            all_texts.append(item['content']['text'])
            for comment in item['comments']:
                all_texts.append(comment['text'])
        return contents, comments_list, labels, all_texts

    def train(self, train_data, val_data=None, epochs=10, batch_size=32, lr=0.001):
        """训练模型"""
        train_contents, train_comments, train_labels, all_texts = self.preprocess_data(train_data)
        if not self.vocab:
            self.vocab = self.build_vocab(all_texts)

        # 初始化数据集和模型
        train_dataset = RumorDataset(train_contents, train_comments, train_labels, self.vocab)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        vocab_size = len(self.vocab)
        self.model = RumorLSTM(vocab_size, self.embedding_dim, self.hidden_dim, 2, self.n_layers, self.dropout)
        self.model.to(self.device)

        # 训练配置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 迭代训练
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                content = batch['content'].to(self.device)
                comments = batch['comments'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(content, comments)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            # 验证（如有验证集）
            if val_data:
                val_metrics = self.evaluate(val_data)
                print(f"Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

    def evaluate(self, test_data):
        """评估模型性能（训练时专用）"""
        if not self.model:
            raise ValueError("模型未训练")
        test_contents, test_comments, test_labels, _ = self.preprocess_data(test_data)
        test_dataset = RumorDataset(test_contents, test_comments, test_labels, self.vocab)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                content = batch['content'].to(self.device)
                comments = batch['comments'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(content, comments)
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds)
        }

    def save_model(self, filepath):
        """保存训练好的模型和词汇表"""
        if not self.model or not self.vocab:
            raise ValueError("模型未训练或词汇表未构建")
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'config': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'n_layers': self.n_layers,
                'dropout': self.dropout,
                'vocab_size': len(self.vocab)
            }
        }
        torch.save(save_data, filepath)
        print(f"模型保存至 {filepath}")


# 辅助函数：加载训练数据（从CSV文件）
# 替换 train_rumor_model.py 中原有的 load_training_data 函数

def load_training_data(comments_path, content_path, is_rumor):
    """
    从 weibo_to_csv.py 导出的'格式化CSV'加载训练数据。
    明确处理无表头格式：
      weibo_data_format.csv: content_id, keyword, content_text, reserved, timestamp
      comments_format.csv: comment_id, content_id, comment_name, comment_text, timestamp
    """
    import pandas as pd
    from datetime import datetime

    def parse_ts(val):
        # 既兼容 int(秒级时间戳)，也兼容形如 '2025-10-27 09:30:01' 的字符串
        try:
            # 数字（或数字字符串）当秒级UNIX时间
            return datetime.fromtimestamp(int(val)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            # 其他字符串直接返回
            return str(val)

    # 读取"内容"CSV - 明确无表头
    try:
        df_content = pd.read_csv(content_path, header=None,
                                 names=['content_id', 'keyword', 'content_text', 'reserved', 'timestamp'])
        contents = []
        for _, r in df_content.iterrows():
            contents.append({
                'id': r['content_id'],
                'text': r['content_text'],
                'time': parse_ts(r['timestamp']),
            })
    except Exception as e:
        print(f"读取内容CSV失败: {e}")
        return []

    # 读取"评论"CSV - 明确无表头
    try:
        df_comments = pd.read_csv(comments_path, header=None,
                                  names=['comment_id', 'content_id', 'comment_name', 'comment_text', 'timestamp'])
        wid_comments = {}
        for _, r in df_comments.iterrows():
            wid = r['content_id']
            wid_comments.setdefault(wid, []).append({
                'text': r['comment_text'],
                'time': parse_ts(r['timestamp']),
                'name': r['comment_name'],
            })
    except Exception as e:
        print(f"读取评论CSV失败: {e}")
        wid_comments = {}

    # 组合为训练数据
    training_data = []
    for c in contents:
        training_data.append({
            'content': c,
            'comments': wid_comments.get(c['id'], []),
            'label': 1 if is_rumor else 0
        })

    print(f"成功加载 {len(training_data)} 条训练数据")
    return training_data
