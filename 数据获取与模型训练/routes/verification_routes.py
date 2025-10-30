# routes/verification_routes.py
from flask import Blueprint, request, jsonify
from datetime import datetime
from models import db, RumorData, ProcessLog

verification_bp = Blueprint('verification', __name__)

@verification_bp.route('/verify-rumor', methods=['POST'])
def verify_rumor():
    """人工校验谣言"""
    try:
        data = request.json
        data_id = data.get('id')
        nature = data.get('nature')  # 'rumor' or 'non-rumor'
        processor = data.get('processor', 'admin')  # 默认处理人
        
        if not data_id or nature not in ['rumor', 'non-rumor']:
            return jsonify({'error': '参数错误'}), 400
        
        # 更新数据状态
        rumor_data = RumorData.query.get(data_id)
        if not rumor_data:
            return jsonify({'error': '数据不存在'}), 404
        
        # 记录修改前的状态
        old_nature = rumor_data.nature
        
        # 更新数据
        rumor_data.nature = nature
        rumor_data.status = 'processed'
        rumor_data.process_time = datetime.utcnow()
        rumor_data.processor = processor
        
        # 记录操作日志
        log = ProcessLog(
            data_id=data_id,
            operation='verify',
            old_nature=old_nature,
            new_nature=nature,
            operator=processor
        )
        
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'message': '校验成功',
            'data': rumor_data.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'校验失败: {str(e)}'}), 500

@verification_bp.route('/modify-verification', methods=['POST'])
def modify_verification():
    """修改校验结果"""
    try:
        data = request.json
        data_id = data.get('id')
        new_nature = data.get('nature')
        processor = data.get('processor', 'admin')
        
        rumor_data = RumorData.query.get(data_id)
        if not rumor_data:
            return jsonify({'error': '数据不存在'}), 404
        
        # 记录修改前的状态
        old_nature = rumor_data.nature
        
        # 更新数据
        rumor_data.nature = new_nature
        rumor_data.process_time = datetime.utcnow()
        rumor_data.processor = processor
        
        # 记录操作日志
        log = ProcessLog(
            data_id=data_id,
            operation='modify',
            old_nature=old_nature,
            new_nature=new_nature,
            operator=processor
        )
        
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'message': '修改成功',
            'data': rumor_data.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'修改失败: {str(e)}'}), 500

@verification_bp.route('/processed-data', methods=['GET'])
def get_processed_data():
    """获取已处理数据"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # 查询已处理的数据，按处理时间降序排列
    data = RumorData.query.filter_by(status='processed')\
        .order_by(RumorData.process_time.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'data': [item.to_dict() for item in data.items],
        'total': data.total,
        'page': page,
        'per_page': per_page,
        'pages': data.pages
    })

@verification_bp.route('/unprocessed-data', methods=['GET'])
def get_unprocessed_data():
    """获取未处理数据"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    data = RumorData.query.filter_by(status='unprocessed')\
        .order_by(RumorData.timestamp.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'data': [item.to_dict() for item in data.items],
        'total': data.total,
        'page': page,
        'per_page': per_page,
        'pages': data.pages
    })

@verification_bp.route('/high-probability-data', methods=['GET'])
def get_high_probability_data():
    """获取高概率谣言数据"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    threshold = request.args.get('threshold', 80, type=float)
    
    data = RumorData.query\
        .filter(RumorData.rumor_probability >= threshold)\
        .filter_by(status='unprocessed')\
        .order_by(RumorData.rumor_probability.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'data': [item.to_dict() for item in data.items],
        'total': data.total,
        'page': page,
        'per_page': per_page,
        'pages': data.pages,
        'threshold': threshold
    })

@verification_bp.route('/verification-stats', methods=['GET'])
def get_verification_stats():
    """获取校验统计信息"""
    try:
        # 统计数据
        total_data = RumorData.query.count()
        processed_data = RumorData.query.filter_by(status='processed').count()
        unprocessed_data = RumorData.query.filter_by(status='unprocessed').count()
        
        # 按性质统计
        rumor_count = RumorData.query.filter_by(nature='rumor').count()
        non_rumor_count = RumorData.query.filter_by(nature='non-rumor').count()
        
        # 按概率区间统计
        high_prob = RumorData.query.filter(RumorData.rumor_probability >= 90).count()
        medium_prob = RumorData.query.filter(
            RumorData.rumor_probability >= 60,
            RumorData.rumor_probability < 90
        ).count()
        low_prob = RumorData.query.filter(RumorData.rumor_probability < 60).count()
        
        return jsonify({
            'total_data': total_data,
            'processed_data': processed_data,
            'unprocessed_data': unprocessed_data,
            'rumor_count': rumor_count,
            'non_rumor_count': non_rumor_count,
            'high_probability': high_prob,
            'medium_probability': medium_prob,
            'low_probability': low_prob,
            'processing_rate': round(processed_data / total_data * 100, 2) if total_data > 0 else 0
        })
        
    except Exception as e:
        return jsonify({'error': f'获取统计信息失败: {str(e)}'}), 500

@verification_bp.route('/process-logs/<int:data_id>', methods=['GET'])
def get_process_logs(data_id):
    """获取处理日志"""
    try:
        logs = ProcessLog.query.filter_by(data_id=data_id)\
            .order_by(ProcessLog.operation_time.desc())\
            .all()
        
        return jsonify({
            'logs': [log.to_dict() for log in logs]
        })
        
    except Exception as e:
        return jsonify({'error': f'获取处理日志失败: {str(e)}'}), 500

@verification_bp.route('/batch-verify', methods=['POST'])
def batch_verify():
    """批量校验"""
    try:
        data = request.json
        items = data.get('items', [])  # [{'id': 1, 'nature': 'rumor'}, ...]
        processor = data.get('processor', 'admin')
        
        if not items:
            return jsonify({'error': '没有要处理的项目'}), 400
        
        processed_count = 0
        errors = []
        
        for item in items:
            try:
                data_id = item.get('id')
                nature = item.get('nature')
                
                if not data_id or nature not in ['rumor', 'non-rumor']:
                    errors.append(f'项目 {data_id} 参数错误')
                    continue
                
                rumor_data = RumorData.query.get(data_id)
                if not rumor_data:
                    errors.append(f'项目 {data_id} 不存在')
                    continue
                
                # 更新数据
                rumor_data.nature = nature
                rumor_data.status = 'processed'
                rumor_data.process_time = datetime.utcnow()
                rumor_data.processor = processor
                
                processed_count += 1
                
            except Exception as e:
                errors.append(f'处理项目 {item.get("id")} 失败: {str(e)}')
        
        db.session.commit()
        
        return jsonify({
            'message': f'批量处理完成，成功处理 {processed_count} 项',
            'processed_count': processed_count,
            'errors': errors
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'批量处理失败: {str(e)}'}), 500

