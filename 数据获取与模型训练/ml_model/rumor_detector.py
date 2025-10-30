# ml_model/rumor_detector.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib
import jieba
import re
import os
from datetime import datetime
import json


class RumorDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.tokenizer = None
        self.max_features = 5000
        self.max_sequence_length = 200
        self.model_path = 'models/'
        
        # 确保模型目录存在
        os.makedirs(self.model_path, exist_ok=True)

    def preprocess_text(self, text):
        """文本预处理"""
        if not text:
            return ''
        
        # 清理文本
        text = re.sub(r'http\S+', '', text)  # 移除URL
        text = re.sub(r'@\w+', '', text)     # 移除@用户名
        text = re.sub(r'#\w+', '', text)     # 移除#标签
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)  # 只保留中文、英文、数字
        
        # 中文分词
        words = jieba.cut(text)
        return ' '.join([word for word in words if len(word) > 1])

    def build_model(self, vocab_size):
        """构建RNN模型"""
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=128, input_length=self.max_sequence_length),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def prepare_sequence_data(self, texts, replies_data=None):
        """准备序列数据，包括主文本和按时间排序的回复"""
        processed_sequences = []
        
        for i, text in enumerate(texts):
            # 处理主文本
            main_text = self.preprocess_text(text)
            
            # 获取对应的回复数据
            replies = []
            if replies_data and i < len(replies_data):
                replies = replies_data[i]
            
            # 按时间排序回复
            if replies:
                sorted_replies = sorted(replies, key=lambda x: x.get('timestamp', ''))
                reply_texts = [self.preprocess_text(reply.get('text', '')) for reply in sorted_replies]
                # 组合主文本和回复
                full_text = main_text + ' ' + ' '.join(reply_texts)
            else:
                full_text = main_text
            
            processed_sequences.append(full_text)
        
        return processed_sequences

    def train(self, texts, labels, replies_data=None):
        """训练模型"""
        print("开始训练模型...")
        
        # 准备序列数据
        processed_texts = self.prepare_sequence_data(texts, replies_data)
        
        # 初始化tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(processed_texts)
        
        # 转换为序列
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # 构建并训练模型
        vocab_size = len(self.tokenizer.word_index) + 1
        self.model = self.build_model(vocab_size)
        
        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, labels, test_size=0.2, random_state=42
        )

        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=10,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # 保存模型和tokenizer
        self.model.save(os.path.join(self.model_path, 'rumor_detector.h5'))
        with open(os.path.join(self.model_path, 'tokenizer.json'), 'w', encoding='utf-8') as f:
            json.dump(self.tokenizer.to_json(), f, ensure_ascii=False)
        
        print("模型训练完成并已保存")
        return history

    def load_model(self):
        """加载已训练的模型"""
        try:
            model_path = os.path.join(self.model_path, 'rumor_detector.h5')
            tokenizer_path = os.path.join(self.model_path, 'tokenizer.json')
            
            if os.path.exists(model_path) and os.path.exists(tokenizer_path):
                self.model = tf.keras.models.load_model(model_path)
                
                with open(tokenizer_path, 'r', encoding='utf-8') as f:
                    tokenizer_json = json.load(f)
                    self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
                
                print("模型加载成功")
                return True
            else:
                print("模型文件不存在")
                return False
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

    def predict(self, text, replies=None):
        """预测谣言概率"""
        if not self.model or not self.tokenizer:
            if not self.load_model():
                return 0.5  # 默认概率

        # 准备数据
        processed_texts = self.prepare_sequence_data([text], [replies] if replies else None)
        
        # 转换为序列
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # 预测
        probability = self.model.predict(padded_sequences)[0][0]
        return float(probability)

    def batch_predict(self, texts, replies_data=None):
        """批量预测"""
        if not self.model or not self.tokenizer:
            if not self.load_model():
                return [0.5] * len(texts)

        # 准备数据
        processed_texts = self.prepare_sequence_data(texts, replies_data)
        
        # 转换为序列
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # 预测
        probabilities = self.model.predict(padded_sequences)
        return [float(p[0]) for p in probabilities]

    def get_feature_importance(self, text):
        """获取特征重要性（简化版）"""
        if not self.model or not self.tokenizer:
            return {}
        
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        # 简单的词频统计
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        return dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])

    def evaluate_model(self, test_texts, test_labels, test_replies=None):
        """评估模型性能"""
        if not self.model or not self.tokenizer:
            return None
        
        processed_texts = self.prepare_sequence_data(test_texts, test_replies)
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # 预测
        predictions = self.model.predict(padded_sequences)
        predicted_labels = (predictions > 0.5).astype(int).flatten()
        
        # 计算准确率
        accuracy = np.mean(predicted_labels == test_labels)
        
        return {
            'accuracy': accuracy,
            'predictions': predictions.flatten().tolist(),
            'predicted_labels': predicted_labels.tolist()
        }

