import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import jieba
import re
from collections import Counter
import json
from datetime import datetime
import os
import sys

# 设置标准输出编码为UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # 对于旧版本Python，使用其他方法
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

torch.manual_seed(42)
np.random.seed(42)

# 其余代码保持不变...

class RumorDataset(Dataset):
    def __init__(self, contents, comments_list, labels, vocab, max_content_len=200, max_comment_len=50, max_comments=100):
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
        """将文本转换为序列"""
        words = jieba.lcut(text)
        sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        return sequence
    
    def pad_sequence(self, sequence, max_len):
        """填充序列"""
        if len(sequence) > max_len:
            return sequence[:max_len]
        else:
            return sequence + [self.vocab['<PAD>']] * (max_len - len(sequence))
    
    def __getitem__(self, idx):
        content_text = self.contents[idx]['text']
        content_seq = self.text_to_sequence(content_text)
        content_padded = self.pad_sequence(content_seq, self.max_content_len)
        
        comments = self.comments_list[idx]
        comments_processed = []
        
        for comment in comments[:self.max_comments]:
            comment_text = comment['text']
            comment_seq = self.text_to_sequence(comment_text)
            comment_padded = self.pad_sequence(comment_seq, self.max_comment_len)
            comments_processed.append(comment_padded)
        
        while len(comments_processed) < self.max_comments:
            comments_processed.append([self.vocab['<PAD>']] * self.max_comment_len)
        
        return {
            'content': torch.tensor(content_padded, dtype=torch.long),
            'comments': torch.tensor(comments_processed, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

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
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, content, comments):
        batch_size = content.size(0)
        num_comments = comments.size(1)
        
        content_embedded = self.embedding(content)
        content_output, (content_hidden, _) = self.content_lstm(content_embedded)
        
        content_output = content_output.transpose(0, 1)  # (seq_len, batch, features)
        content_attn_output, _ = self.content_attention(content_output, content_output, content_output)
        content_attn_output = content_attn_output.transpose(0, 1)  # (batch, seq_len, features)
        
        content_pooled = torch.mean(content_attn_output, dim=1)  # (batch, features)
        
        comments_embedded = self.embedding(comments)  # (batch, num_comments, comment_len, embedding_dim)
        
        comments_reshaped = comments_embedded.view(batch_size * num_comments, -1, comments_embedded.size(-1))
        comments_output, (comments_hidden, _) = self.comment_lstm(comments_reshaped)
        
        comments_output = comments_output.transpose(0, 1)
        comments_attn_output, _ = self.comment_attention(comments_output, comments_output, comments_output)
        comments_attn_output = comments_attn_output.transpose(0, 1)
        
        comments_pooled = torch.mean(comments_attn_output, dim=1)  # (batch*num_comments, features)
        comments_pooled = comments_pooled.view(batch_size, num_comments, -1)  # (batch, num_comments, features)
        
        comments_aggregated = torch.mean(comments_pooled, dim=1)  # (batch, features)
        
        combined = torch.cat([content_pooled, comments_aggregated], dim=1)  # (batch, features*2)
        
        output = self.fc(combined)
        
        return output

class RumorDetector:
    def __init__(self, embedding_dim=128, hidden_dim=64, n_layers=2, dropout=0.3):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.vocab = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_vocab(self, texts, min_freq=2):
        """构建词汇表"""
        word_counter = Counter()
        
        for text in texts:
            words = jieba.lcut(text)
            word_counter.update(words)
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        
        for word, count in word_counter.items():
            if count >= min_freq:
                vocab[word] = idx
                idx += 1
        
        return vocab
    
    def preprocess_data(self, data):
        """预处理数据"""
        contents = []
        comments_list = []
        labels = []
        all_texts = []
        
        for item in data:
            content = item['content']
            comments = item['comments']
            
            contents.append(content)
            comments_list.append(comments)
            labels.append(item['label'])
            
            all_texts.append(content['text'])
            for comment in comments:
                all_texts.append(comment['text'])
        
        return contents, comments_list, labels, all_texts
    
    def train(self, train_data, val_data=None, epochs=10, batch_size=32, learning_rate=0.001):
        """训练模型"""
        train_contents, train_comments, train_labels, all_texts = self.preprocess_data(train_data)
        
        if self.vocab is None:
            self.vocab = self.build_vocab(all_texts)
        
        train_dataset = RumorDataset(train_contents, train_comments, train_labels, self.vocab)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        vocab_size = len(self.vocab)
        self.model = RumorLSTM(vocab_size, self.embedding_dim, self.hidden_dim, 
                              output_dim=2, n_layers=self.n_layers, dropout=self.dropout)
        self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
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
            
            if val_data:
                val_metrics = self.evaluate(val_data)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, '
                      f'Val Acc: {val_metrics["accuracy"]:.4f}, Val F1: {val_metrics["f1"]:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    def evaluate(self, test_data):
        """评估模型"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        test_contents, test_comments, test_labels, _ = self.preprocess_data(test_data)
        test_dataset = RumorDataset(test_contents, test_comments, test_labels, self.vocab)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                content = batch['content'].to(self.device)
                comments = batch['comments'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(content, comments)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict(self, content, comments):
        """预测单条数据"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        
        content_seq = self._text_to_sequence(content['text'])
        content_padded = self._pad_sequence(content_seq, 200)
        
        comments_processed = []
        for comment in comments[:100]:
            comment_seq = self._text_to_sequence(comment['text'])
            comment_padded = self._pad_sequence(comment_seq, 50)
            comments_processed.append(comment_padded)
        
        while len(comments_processed) < 100:
            comments_processed.append([self.vocab['<PAD>']] * 50)
        
        content_tensor = torch.tensor([content_padded], dtype=torch.long).to(self.device)
        comments_tensor = torch.tensor([comments_processed], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.model(content_tensor, comments_tensor)
            prediction = torch.argmax(output, dim=1)
            probability = torch.softmax(output, dim=1)
        
        return {
            'prediction': '谣言' if prediction.item() == 1 else '非谣言',
            'confidence': probability[0][prediction.item()].item()
        }
    
    def _text_to_sequence(self, text):
        words = jieba.lcut(text)
        sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        return sequence
    
    def _pad_sequence(self, sequence, max_len):
        if len(sequence) > max_len:
            return sequence[:max_len]
        else:
            return sequence + [self.vocab['<PAD>']] * (max_len - len(sequence))
    def save(self, filepath):
        if self.model is None:
            raise ValueError("请先创建模型!")
        
        if self.vocab is None:
            raise ValueError("Vocabulary not built yet! Cannot save model without vocabulary.")

        save_data = {
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'model_config': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'n_layers': self.n_layers,
                'dropout': self.dropout,
                'vocab_size': len(self.vocab)
            }
        }
        
        torch.save(save_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        save_data = torch.load(filepath, map_location=self.device)
        
        self.vocab = save_data['vocab']
        
        model_config = save_data['model_config']
        self.embedding_dim = model_config['embedding_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        
        vocab_size = model_config['vocab_size']
        self.model = RumorLSTM(vocab_size, self.embedding_dim, self.hidden_dim,output_dim=2, n_layers=self.n_layers, dropout=self.dropout)
        
        self.model.load_state_dict(save_data['model_state_dict'])
        self.model.to(self.device)

def create_sample_data():
    """创建示例数据"""
    sample_data = [
        {
            'content': {
                'text': '今天发生了一件重大事件，据说有外星人降临地球！',
                'time': '2024-01-01 10:00:00'
            },
            'comments': [
                {'text': '真的吗？太不可思议了！', 'time': '2024-01-01 10:05:00', 'name': '用户A', 'reply2': ''},
                {'text': '这是谣言，不要相信', 'time': '2024-01-01 10:10:00', 'name': '用户B', 'reply2': ''},
                {'text': '我也听说了，但还没有官方证实', 'time': '2024-01-01 10:15:00', 'name': '用户C', 'reply2': '用户A'}
            ],
            'label': 1  # 1表示谣言，0表示非谣言
        },
        {
            'content': {
                'text': '市政府发布通知，明天全市停水检修',
                'time': '2024-01-02 09:00:00'
            },
            'comments': [
                {'text': '收到，谢谢提醒', 'time': '2024-01-02 09:05:00', 'name': '用户D', 'reply2': ''},
                {'text': '官方已经确认了这个消息', 'time': '2024-01-02 09:10:00', 'name': '用户E', 'reply2': ''}
            ],
            'label': 0  # 非谣言
        }
    ]
    return sample_data

def download(comments_filepath,content_filepath, rumor = False):
    '''下载对应谣言(或非谣言)数据集'''
    contents = []
    df = pd.read_csv(content_filepath, header=None)
    first_line = True
    for index, row in df.iterrows():
        if first_line:
            first_line = False
            continue
        content = {}
        content['text'] = row[2]
        content['id'] = row[0]
        timestamp = int(row[4])
        dt_object = datetime.fromtimestamp(timestamp)
        formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
        content['time'] = formatted_time
        contents.append(content)
    comments = []
    df = pd.read_csv(comments_filepath, header=None)
    first_line = True
    wid_cmts = {}
    for index, row in df.iterrows():
        if first_line:
            first_line = False
            continue
        if not row[1] in wid_cmts:
            wid_cmts[row[1]] = []
        comment = {}
        comment['text'] = row[3]
        timestamp = int(row[4])
        dt_object = datetime.fromtimestamp(timestamp)
        formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
        comment['time'] = formatted_time
        comment['name'] = row[2]
        wid_cmts[row[1]].append(comment)
    for content in contents:
        comments.append(wid_cmts[content['id']])
    datas = []
    for i in range(len(contents)):
        data = {}
        content = contents[i]
        del content['id']
        data['content'] = content
        data['comments'] = comments[i]
        data['label'] = 1 if rumor else 0
        datas.append(data)
    return datas
