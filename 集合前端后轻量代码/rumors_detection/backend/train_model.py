from rumor_detect_model import download,train_test_split,RumorDetector
if __name__ == "__main__":
    data = []
    for i in range(1,16):
        comments_filepath = f'./dataset/train_output_is_rumor/{i}/comments.csv'
        content_filepath = f'./dataset/train_output_is_rumor/{i}/weibo_data.csv'
        print(f'rumors:{i}')
        rumor_data = download(comments_filepath,content_filepath,rumor = True)
        data += rumor_data
        if i < 9:
            print(f'not_rumors:{i}')
            comments_filepath = f'./dataset/train_output_not_rumor/{i}/comments.csv'
            content_filepath = f'./dataset/train_output_not_rumor/{i}/weibo_data.csv'
            n_rumor_data = download(comments_filepath,content_filepath,rumor = False)
            data += n_rumor_data
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    detector = RumorDetector()
    
    print("开始训练模型...")
    detector.train(train_data, epochs=1, batch_size=10, learning_rate=0.001)
    
    print("\n评估模型...")
    metrics = detector.evaluate(test_data)
    print(f"测试集结果: 准确率: {metrics['accuracy']:.4f}, F1分数: {metrics['f1']:.4f}")
    filepath = './model/model.pth'
    detector.save(filepath)
    print(f"成功保存至{filepath}")