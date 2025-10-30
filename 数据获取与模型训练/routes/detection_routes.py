# routes/detection_routes.py
from flask import Blueprint, request, jsonify
from ml_model.rumor_detector import RumorDetector
import redis
import json
import tensorflow as tf

detection_bp = Blueprint('detection', __name__)
detector = RumorDetector()

# Redis缓存配置
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

@detection_bp.route('/predict_rumor', methods=['POST'])
def predict_rumor():
    try:
        data = request.json
        text = data.get('text', '')
        replies = data.get('replies', [])
        data_id = data.get('data_id')
        
        if not text:
            return jsonify({'error': '文本内容不能为空'}), 400
        
        # 检查缓存
        cache_key = f"rumor_prediction:{hash(text + str(replies))}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            return jsonify(json.loads(cached_result))
        
        # 预测谣言概率
        probability = detector.predict(text, replies)
        probability_percent = round(probability * 100, 2)
        
        # 获取特征重要性
        feature_importance = detector.get_feature_importance(text)
        
        result = {
            'probability': probability_percent,
            'risk_level': get_risk_level(probability_percent),
            'text': text[:100] + '...' if len(text) > 100 else text,
            'feature_importance': feature_importance,
            'timestamp': tf.keras.utils.get_file('timestamp', 'http://example.com').split('/')[-1] if hasattr(tf.keras.utils, 'get_file') else None
        }
        
        # 缓存结果（5分钟过期）
        redis_client.setex(cache_key, 300, json.dumps(result))
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@detection_bp.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        texts = data.get('texts', [])
        replies_data = data.get('replies_data', [])
        
        if not texts:
            return jsonify({'error': '文本列表不能为空'}), 400
        
        # 批量预测
        probabilities = detector.batch_predict(texts, replies_data)
        
        results = []
        for i, (text, prob) in enumerate(zip(texts, probabilities)):
            prob_percent = round(prob * 100, 2)
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'probability': prob_percent,
                'risk_level': get_risk_level(prob_percent)
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': f'批量预测失败: {str(e)}'}), 500

@detection_bp.route('/train_model', methods=['POST'])
def train_model():
    try:
        data = request.json
        texts = data.get('texts', [])
        labels = data.get('labels', [])
        replies_data = data.get('replies_data', [])
        
        if len(texts) != len(labels):
            return jsonify({'error': '文本和标签数量不匹配'}), 400
        
        if len(texts) < 10:
            return jsonify({'error': '训练数据太少，至少需要10条'}), 400
        
        # 训练模型
        history = detector.train(texts, labels, replies_data)
        
        return jsonify({
            'message': '模型训练完成',
            'training_samples': len(texts),
            'history': {
                'loss': history.history['loss'][-1],
                'accuracy': history.history['accuracy'][-1],
                'val_loss': history.history['val_loss'][-1],
                'val_accuracy': history.history['val_accuracy'][-1]
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'训练失败: {str(e)}'}), 500

@detection_bp.route('/evaluate_model', methods=['POST'])
def evaluate_model():
    try:
        data = request.json
        test_texts = data.get('test_texts', [])
        test_labels = data.get('test_labels', [])
        test_replies = data.get('test_replies', [])
        
        if len(test_texts) != len(test_labels):
            return jsonify({'error': '测试文本和标签数量不匹配'}), 400
        
        # 评估模型
        evaluation = detector.evaluate_model(test_texts, test_labels, test_replies)
        
        if evaluation is None:
            return jsonify({'error': '模型未加载'}), 400
        
        return jsonify(evaluation)
        
    except Exception as e:
        return jsonify({'error': f'评估失败: {str(e)}'}), 500

@detection_bp.route('/model_status', methods=['GET'])
def model_status():
    try:
        model_loaded = detector.model is not None and detector.tokenizer is not None
        
        return jsonify({
            'model_loaded': model_loaded,
            'max_features': detector.max_features,
            'max_sequence_length': detector.max_sequence_length,
            'model_path': detector.model_path
        })
        
    except Exception as e:
        return jsonify({'error': f'获取模型状态失败: {str(e)}'}), 500

def get_risk_level(probability):
    """根据概率确定风险等级"""
    if probability >= 90:
        return 'high'
    elif probability >= 60:
        return 'medium'
    else:
        return 'low'

