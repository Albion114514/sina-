from rumor_detect_model import RumorDetector
if __name__ == "__main__":
    detector = RumorDetector()
    filepath = './model.pth'
    detector.load(filepath)
    print(f"成功从{filepath}下载模型")
    print("\n预测新数据...")
    new_content = {'text': '最新消息：明天会有特大暴雨，建议居家办公', 'time': '2024-01-03 08:00:00'}
    new_comments = [
        {'text': '气象局已经辟谣了，这是假消息', 'time': '2024-01-03 08:05:00', 'name': '用户F'},
        {'text': '不要传播不实信息', 'time': '2024-01-03 08:10:00', 'name': '用户G'}
    ]
    
    result = detector.predict(new_content, new_comments)
    print(f"预测结果: {result['prediction']}, 置信度: {result['confidence']:.4f}")