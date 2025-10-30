import torch
import jieba
import os


# 复用模型结构（必须与训练组件中的RumorLSTM完全一致）
class RumorLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(RumorLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.content_lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers,
                                          batch_first=True, dropout=dropout, bidirectional=True)
        self.comment_lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers,
                                          batch_first=True, dropout=dropout, bidirectional=True)
        self.content_attention = torch.nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.comment_attention = torch.nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim)
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


# 预测器类
class RumorPredictor:
    def __init__(self):
        self.model = None
        self.vocab = None
        self.config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_path):
        """加载训练好的模型和词汇表"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

        save_data = torch.load(model_path, map_location=self.device)
        self.vocab = save_data['vocab']
        self.config = save_data['config']

        # 重建模型
        self.model = RumorLSTM(
            vocab_size=self.config['vocab_size'],
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=2,
            n_layers=self.config['n_layers'],
            dropout=self.config['dropout']
        )
        self.model.load_state_dict(save_data['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()  # 切换至评估模式
        print(f"模型加载成功：{model_path}")

    def _text_to_sequence(self, text):
        """文本转序列（内部辅助方法）"""
        words = jieba.lcut(text)
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in words]

    def _pad_sequence(self, sequence, max_len):
        """序列填充（内部辅助方法）"""
        if len(sequence) > max_len:
            return sequence[:max_len]
        return sequence + [self.vocab['<PAD>']] * (max_len - len(sequence))

    def predict(self, content_text, comments):
        """
        预测单条内容是否为谣言
        :param content_text: 内容文本（字符串）
        :param comments: 评论列表（每个元素为{'text': 评论内容}）
        :return: 预测结果字典（含标签和置信度）
        """
        if not self.model or not self.vocab:
            raise ValueError("请先加载模型（load_model）")

        # 处理内容文本
        content_seq = self._text_to_sequence(content_text)
        content_padded = self._pad_sequence(content_seq, max_len=200)  # 与训练时一致
        content_tensor = torch.tensor([content_padded], dtype=torch.long).to(self.device)

        # 处理评论
        comments_processed = []
        for comment in comments[:100]:  # 最多取100条评论
            comment_seq = self._text_to_sequence(comment['text'])
            comment_padded = self._pad_sequence(comment_seq, max_len=50)  # 与训练时一致
            comments_processed.append(comment_padded)

        # 不足100条评论用PAD填充
        while len(comments_processed) < 100:
            comments_processed.append([self.vocab['<PAD>']] * 50)

        comments_tensor = torch.tensor([comments_processed], dtype=torch.long).to(self.device)

        # 预测
        with torch.no_grad():
            output = self.model(content_tensor, comments_tensor)
            pred_label = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred_label].item()

        return {
            'prediction': '谣言' if pred_label == 1 else '非谣言',
            'confidence': round(confidence, 4)
        }