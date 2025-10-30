# routes/search_routes.py
from flask import Blueprint, request, jsonify
from models import db, RumorData
from collections import Counter
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import redis
import os

search_bp = Blueprint('search', __name__)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

@search_bp.route('/wordcloud', methods=['GET'])
def generate_wordcloud():
    """生成词云图"""
    try:
        # 检查缓存
        cache_key = "wordcloud:latest"
        cached_image = redis_client.get(cache_key)
        
        if cached_image:
            return jsonify({
                'image': cached_image
            })
        
        # 从数据库获取所有文本
        texts = RumorData.query.with_entities(RumorData.text).all()
        all_text = ' '.join([item[0] for item in texts if item[0]])
        
        if not all_text:
            return jsonify({'error': '没有数据生成词云'}), 400
        
        # 中文分词
        words = jieba.cut(all_text)
        
        # 过滤停用词
        stop_words = {'的', '了', '是', '在', '和', '有', '我', '你', '他', '她', '它', '这', '那', '一个', '这个', '那个', '什么', '怎么', '为什么', '因为', '所以', '但是', '然后', '现在', '今天', '昨天', '明天'}
        filtered_words = [word for word in words if len(word) > 1 and word not in stop_words]
        
        # 统计词频
        word_freq = Counter(filtered_words)
        
        # 生成词云
        wc = WordCloud(
            font_path='fonts/simhei.ttf' if os.path.exists('fonts/simhei.ttf') else None,
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        )
        
        wordcloud = wc.generate_from_frequencies(word_freq)
        
        # 转换为base64
        img_buffer = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        # 缓存结果（1天过期）
        redis_client.setex(cache_key, 86400, img_base64)
        
        return jsonify({
            'image': img_base64
        })
        
    except Exception as e:
        return jsonify({'error': f'生成词云失败: {str(e)}'}), 500

@search_bp.route('/filter-by-keyword', methods=['POST'])
def filter_by_keyword():
    """根据关键词筛选数据"""
    try:
        data = request.json
        keyword = data.get('keyword', '')
        probability_threshold = data.get('probability_threshold', 0)
        data_type = data.get('type', 'all')  # all, processed, unprocessed
        page = data.get('page', 1)
        per_page = data.get('per_page', 20)
        
        if not keyword:
            return jsonify({'error': '关键词不能为空'}), 400
        
        # 构建查询
        query = RumorData.query.filter(RumorData.text.like(f'%{keyword}%'))
        
        # 根据数据类型过滤
        if data_type == 'processed':
            query = query.filter_by(status='processed')
        elif data_type == 'unprocessed':
            query = query.filter_by(status='unprocessed')
        
        # 根据概率阈值过滤
        if probability_threshold > 0:
            query = query.filter(RumorData.rumor_probability >= probability_threshold)
        
        # 分页查询
        paginated_query = query.order_by(RumorData.timestamp.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'data': [item.to_dict() for item in paginated_query.items],
            'total': paginated_query.total,
            'page': page,
            'per_page': per_page,
            'pages': paginated_query.pages
        })
        
    except Exception as e:
        return jsonify({'error': f'筛选失败: {str(e)}'}), 500

@search_bp.route('/hot-keywords', methods=['GET'])
def get_hot_keywords():
    """获取热门关键词"""
    try:
        # 从数据库获取最近的数据
        recent_texts = RumorData.query\
            .order_by(RumorData.timestamp.desc())\
            .limit(1000)\
            .with_entities(RumorData.text)\
            .all()
        
        all_text = ' '.join([item[0] for item in recent_texts if item[0]])
        
        if not all_text:
            return jsonify({'keywords': []})
        
        # 分词和统计
        words = jieba.cut(all_text)
        stop_words = {'的', '了', '是', '在', '和', '有', '我', '你', '他', '她', '它', '这', '那', '一个', '这个', '那个', '什么', '怎么', '为什么', '因为', '所以', '但是', '然后', '现在', '今天', '昨天', '明天'}
        filtered_words = [word for word in words if len(word) > 1 and word not in stop_words]
        
        word_freq = Counter(filtered_words)
        top_keywords = [{'word': word, 'count': count} for word, count in word_freq.most_common(20)]
        
        return jsonify({
            'keywords': top_keywords
        })
        
    except Exception as e:
        return jsonify({'error': f'获取热门关键词失败: {str(e)}'}), 500

@search_bp.route('/search-suggestions', methods=['GET'])
def get_search_suggestions():
    """获取搜索建议"""
    try:
        keyword = request.args.get('keyword', '')
        
        if not keyword:
            return jsonify({'suggestions': []})
        
        # 从数据库搜索相关关键词
        suggestions = RumorData.query\
            .filter(RumorData.text.like(f'%{keyword}%'))\
            .with_entities(RumorData.text)\
            .limit(10)\
            .all()
        
        # 提取关键词
        suggestion_words = set()
        for item in suggestions:
            words = jieba.cut(item[0])
            for word in words:
                if keyword in word and len(word) > len(keyword):
                    suggestion_words.add(word)
        
        return jsonify({
            'suggestions': list(suggestion_words)[:10]
        })
        
    except Exception as e:
        return jsonify({'error': f'获取搜索建议失败: {str(e)}'}), 500

@search_bp.route('/wordcloud-cache', methods=['DELETE'])
def clear_wordcloud_cache():
    """清除词云缓存"""
    try:
        redis_client.delete("wordcloud:latest")
        return jsonify({'message': '词云缓存已清除'})
    except Exception as e:
        return jsonify({'error': f'清除缓存失败: {str(e)}'}), 500

