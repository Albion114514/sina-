# app.py - 主应用文件
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

# 导入蓝图
from routes.detection_routes import detection_bp
from routes.search_routes import search_bp
from routes.verification_routes import verification_bp

# app.py（只展示修改处）
from settings import load_config, build_sqlalchemy_uri

app = Flask(__name__)
CORS(app)

# 统一从 config.json 读取
_cfg = load_config()
app.config['SQLALCHEMY_DATABASE_URI'] = build_sqlalchemy_uri(_cfg)  # ← 修复端口错配
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'

db = SQLAlchemy(app)

# 注册蓝图
app.register_blueprint(detection_bp, url_prefix='/api')
app.register_blueprint(search_bp, url_prefix='/api')
app.register_blueprint(verification_bp, url_prefix='/api')

# 在 app.py 的蓝图导入后添加
from rumor_predict_model import RumorPredictionService

# 全局预测服务实例
prediction_service = None


def init_prediction_service():
    """初始化预测服务"""
    global prediction_service
    try:
        prediction_service = RumorPredictionService()
        if prediction_service.load_model() and prediction_service.connect_database():
            print("✅ 预测服务初始化成功")
            return True
        else:
            print("❌ 预测服务初始化失败")
            return False
    except Exception as e:
        print(f"❌ 预测服务初始化异常: {e}")
        return False


# 在应用启动时初始化
@app.before_first_request
def initialize_services():
    init_prediction_service()


# 添加预测路由
@app.route('/api/predict/<string:weibo_id>', methods=['GET'])
def predict_weibo(weibo_id):
    """预测单条微博"""
    if not prediction_service:
        return jsonify({'error': '预测服务未初始化'}), 500

    try:
        result = prediction_service.predict_single_weibo(weibo_id)
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': '预测失败'}), 400
    except Exception as e:
        return jsonify({'error': f'预测异常: {str(e)}'}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """批量预测"""
    if not prediction_service:
        return jsonify({'error': '预测服务未初始化'}), 500

    try:
        data = request.json
        weibo_ids = data.get('weibo_ids', [])
        results = prediction_service.predict_batch_weibos(weibo_ids)
        return jsonify({'results': results, 'total': len(results)})
    except Exception as e:
        return jsonify({'error': f'批量预测异常: {str(e)}'}), 500


@app.route('/api/predict/keyword', methods=['POST'])
def predict_by_keyword():
    """关键字预测"""
    if not prediction_service:
        return jsonify({'error': '预测服务未初始化'}), 500

    try:
        data = request.json
        keyword = data.get('keyword', '')
        limit = data.get('limit', 50)
        # 修正：将 service 改为 prediction_service
        results = prediction_service.predict_by_keyword(keyword, limit)
        return jsonify({'results': results, 'total': len(results)})
    except Exception as e:
        return jsonify({'error': f'关键字预测异常: {str(e)}'}), 500


@app.route('/api/predict/stats', methods=['GET'])
def get_prediction_stats():
    """获取预测统计"""
    if not prediction_service:
        return jsonify({'error': '预测服务未初始化'}), 500

    try:
        stats = prediction_service.get_prediction_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'获取统计异常: {str(e)}'}), 500

# 数据模型
class RumorData(db.Model):
    __tablename__ = 'rumor_data'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    text = db.Column(db.Text, nullable=False)
    image = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    source_platform = db.Column(db.String(50))
    post_id = db.Column(db.String(100), unique=True)
    status = db.Column(db.String(20), default='unprocessed')
    rumor_probability = db.Column(db.Float)
    nature = db.Column(db.String(20))  # 'rumor' or 'non-rumor'
    process_time = db.Column(db.DateTime)
    processor = db.Column(db.String(50))

    # 关系
    replies = db.relationship('Reply', backref='rumor_data', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'image': self.image,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp else None,
            'source_platform': self.source_platform,
            'status': self.status,
            'rumor_probability': self.rumor_probability,
            'nature': self.nature,
            'process_time': self.process_time.strftime('%Y-%m-%d %H:%M:%S') if self.process_time else None,
            'processor': self.processor
        }


class Reply(db.Model):
    __tablename__ = 'replies'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rumor_data_id = db.Column(db.Integer, db.ForeignKey('rumor_data.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.String(100))
    parent_id = db.Column(db.Integer, db.ForeignKey('replies.id'))  # 支持回复的回复

    # 关系
    parent = db.relationship('Reply', remote_side=[id], backref='children')

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp else None,
            'user_id': self.user_id,
            'parent_id': self.parent_id
        }


class ProcessLog(db.Model):
    __tablename__ = 'process_log'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    data_id = db.Column(db.Integer, db.ForeignKey('rumor_data.id'), nullable=False)
    operation = db.Column(db.String(20), nullable=False)  # 'verify' or 'modify'
    old_nature = db.Column(db.String(20))
    new_nature = db.Column(db.String(20), nullable=False)
    operator = db.Column(db.String(50), nullable=False)
    operation_time = db.Column(db.DateTime, default=datetime.utcnow)

    # 关系
    rumor_data = db.relationship('RumorData', backref=db.backref('logs', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'data_id': self.data_id,
            'operation': self.operation,
            'old_nature': self.old_nature,
            'new_nature': self.new_nature,
            'operator': self.operator,
            'operation_time': self.operation_time.strftime('%Y-%m-%d %H:%M:%S')
        }


# 新增微博数据模型（用于移动端爬虫数据）
class WeiboData(db.Model):
    __tablename__ = 'weibo_data'
    id = db.Column(db.String(50), primary_key=True)
    keyword_id = db.Column(db.Integer, db.ForeignKey('keywords.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    pics = db.Column(db.Text)
    timestamp = db.Column(db.BigInteger, nullable=False)
    source = db.Column(db.String(100), default='新浪微博移动端')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 新增移动端字段
    user_name = db.Column(db.String(255), default='')
    reposts_count = db.Column(db.Integer, default=0)
    comments_count = db.Column(db.Integer, default=0)
    attitudes_count = db.Column(db.Integer, default=0)

    # 关系
    keyword = db.relationship('Keyword', backref='weibo_data')
    comments = db.relationship('Comment', backref='weibo', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'keyword': self.keyword.keyword if self.keyword else '',
            'text': self.text,
            'pics': self.pics.split(',') if self.pics else [],
            'timestamp': datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'source': self.source,
            'user_name': self.user_name,
            'reposts_count': self.reposts_count,
            'comments_count': self.comments_count,
            'attitudes_count': self.attitudes_count,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }


class Keyword(db.Model):
    __tablename__ = 'keywords'
    id = db.Column(db.Integer, primary_key=True)
    keyword = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Comment(db.Model):
    __tablename__ = 'comments'
    id = db.Column(db.String(50), primary_key=True)
    weibo_id = db.Column(db.String(50), db.ForeignKey('weibo_data.id'), nullable=False)
    user = db.Column(db.String(100))
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.BigInteger, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'weibo_id': self.weibo_id,
            'user': self.user,
            'content': self.content,
            'timestamp': datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }


# 基础数据接口
@app.route('/api/data', methods=['GET'])
def get_data():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)

    # 按时间降序排列，分页查询
    data = RumorData.query.order_by(RumorData.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        'data': [item.to_dict() for item in data.items],
        'total': data.total,
        'page': page,
        'per_page': per_page,
        'pages': data.pages
    })


# 新增微博数据接口
@app.route('/api/weibo_data', methods=['GET'])
def get_weibo_data():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    keyword = request.args.get('keyword', '')

    # 构建查询
    query = WeiboData.query

    if keyword:
        query = query.join(Keyword).filter(Keyword.keyword.like(f'%{keyword}%'))

    # 按时间降序排列，分页查询
    data = query.order_by(WeiboData.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        'data': [item.to_dict() for item in data.items],
        'total': data.total,
        'page': page,
        'per_page': per_page,
        'pages': data.pages
    })


@app.route('/api/weibo_data/<string:weibo_id>', methods=['GET'])
def get_weibo_detail(weibo_id):
    item = WeiboData.query.get_or_404(weibo_id)
    result = item.to_dict()

    # 获取评论数据
    comments = Comment.query.filter_by(weibo_id=weibo_id).order_by(Comment.timestamp.asc()).all()
    result['comments'] = [comment.to_dict() for comment in comments]

    return jsonify(result)


@app.route('/api/keywords', methods=['GET'])
def get_keywords():
    keywords = Keyword.query.order_by(Keyword.created_at.desc()).all()
    return jsonify([{'id': kw.id, 'keyword': kw.keyword} for kw in keywords])


@app.route('/api/data/<int:data_id>', methods=['GET'])
def get_data_detail(data_id):
    item = RumorData.query.get_or_404(data_id)
    result = item.to_dict()

    # 获取回复数据
    replies = Reply.query.filter_by(rumor_data_id=data_id).order_by(Reply.timestamp.asc()).all()
    result['replies'] = [reply.to_dict() for reply in replies]

    return jsonify(result)


@app.route('/api/data/<int:data_id>/replies', methods=['GET'])
def get_replies(data_id):
    replies = Reply.query.filter_by(rumor_data_id=data_id).order_by(Reply.timestamp.asc()).all()
    return jsonify([reply.to_dict() for reply in replies])


@app.route('/api/data/<int:data_id>/replies', methods=['POST'])
def add_reply(data_id):
    data = request.json
    reply = Reply(
        rumor_data_id=data_id,
        text=data.get('text', ''),
        user_id=data.get('user_id', ''),
        parent_id=data.get('parent_id')
    )

    db.session.add(reply)
    db.session.commit()

    return jsonify(reply.to_dict())


# 健康检查
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)