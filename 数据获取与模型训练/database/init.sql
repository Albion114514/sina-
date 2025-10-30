-- 创建数据库和表
CREATE DATABASE IF NOT EXISTS rumor_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE rumor_db;

-- 谣言数据表
CREATE TABLE IF NOT EXISTS rumor_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL,
    image VARCHAR(500),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    source_platform VARCHAR(50),
    post_id VARCHAR(100) UNIQUE,
    status ENUM('unprocessed', 'processed') DEFAULT 'unprocessed',
    rumor_probability DECIMAL(5,2),
    nature ENUM('rumor', 'non-rumor'),
    process_time DATETIME,
    processor VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 回复数据表
CREATE TABLE IF NOT EXISTS replies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rumor_data_id INT NOT NULL,
    text TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(100),
    parent_id INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (rumor_data_id) REFERENCES rumor_data(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_id) REFERENCES replies(id) ON DELETE CASCADE
);

-- 处理日志表
CREATE TABLE IF NOT EXISTS process_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    data_id INT NOT NULL,
    operation ENUM('verify', 'modify') NOT NULL,
    old_nature ENUM('rumor', 'non-rumor'),
    new_nature ENUM('rumor', 'non-rumor') NOT NULL,
    operator VARCHAR(50) NOT NULL,
    operation_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (data_id) REFERENCES rumor_data(id) ON DELETE CASCADE
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_timestamp ON rumor_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_status ON rumor_data(status);
CREATE INDEX IF NOT EXISTS idx_nature ON rumor_data(nature);
CREATE INDEX IF NOT EXISTS idx_probability ON rumor_data(rumor_probability);
CREATE INDEX IF NOT EXISTS idx_source ON rumor_data(source_platform);
CREATE INDEX IF NOT EXISTS idx_replies_data_id ON replies(rumor_data_id);
CREATE INDEX IF NOT EXISTS idx_replies_timestamp ON replies(timestamp);
CREATE INDEX IF NOT EXISTS idx_log_data_id ON process_log(data_id);
CREATE INDEX IF NOT EXISTS idx_log_operation_time ON process_log(operation_time);

-- 插入示例数据
INSERT INTO rumor_data (text, source_platform, post_id, rumor_probability, status) VALUES
('原来是真事啊?![吃惊]新闻都报道了!!4岁成都男孩划伤宝马，宝马车主打了小孩儿一耳光，奶奶叫来6辆奔驰砸毁宝马!!![恐怖]然后爸爸来了把车买下，并且砸烂!!![衰]', '新浪微博', 'weibo_001', 99.90, 'unprocessed'),
('若是有人在路上拦你，向你推销福建...', '新浪微博', 'weibo_002', 99.89, 'unprocessed'),
('《以下食物三小时之内不宜同时食用》:...', '新浪微博', 'weibo_003', 98.61, 'unprocessed'),
('今天约好去海底捞，结果发现人特别多，排了2小时队才吃到', '新浪微博', 'weibo_004', 15.20, 'processed'),
('刘翔在奥运会上表现很出色，为中国争光', '新浪微博', 'weibo_005', 25.30, 'processed');

-- 插入示例回复数据
INSERT INTO replies (rumor_data_id, text, user_id) VALUES
(1, '真的假的？这也太夸张了吧', 'user_001'),
(1, '我也看到了这个新闻，确实很震惊', 'user_002'),
(1, '这种新闻一般都是假的，不要相信', 'user_003'),
(2, '福建什么？怎么不说完', 'user_004'),
(2, '我也遇到过类似的推销', 'user_005'),
(3, '这个我知道，确实不能一起吃', 'user_006'),
(4, '海底捞确实很火，排队是常态', 'user_007'),
(4, '我也经常去海底捞，味道不错', 'user_008');

-- 插入示例处理日志
INSERT INTO process_log (data_id, operation, old_nature, new_nature, operator) VALUES
(4, 'verify', NULL, 'non-rumor', 'admin'),
(5, 'verify', NULL, 'non-rumor', 'admin');

-- 更新已处理数据的性质
UPDATE rumor_data SET nature = 'non-rumor', process_time = NOW(), processor = 'admin' WHERE id IN (4, 5);

