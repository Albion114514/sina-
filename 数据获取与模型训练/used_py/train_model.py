from rumor_detect_model import download,train_test_split,RumorDetector
if __name__ == "__main__":
    comments_filepath = './rumor_comments.csv'
    content_filepath = './rumor_weibo_data.csv'
    data = download(comments_filepath,content_filepath,rumor = True)
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    detector = RumorDetector()
    
    print("开始训练模型...")
    detector.train(train_data, epochs=5, batch_size=2, learning_rate=0.001)
    
    print("\n评估模型...")
    metrics = detector.evaluate(test_data)
    print(f"测试集结果: 准确率: {metrics['accuracy']:.4f}, F1分数: {metrics['f1']:.4f}")
    filepath = './model.pth'
    detector.save(filepath)
    print(f"成功保存至{filepath}")