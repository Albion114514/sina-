# 谣言检测系统

基于深度学习的社交媒体谣言检测系统，支持数据展示、自动检测、人工校验和关键词搜索等功能。

## 功能特性

### 1. 数据展示模块
- 支持HTML和JSON格式数据源
- 滚动式加载展示社交媒体数据
- 按时间排序，支持内容摘要显示
- 详情查看功能

### 2. 服务器检测结果呈现模块
- 基于TF-IDF和RNN的深度学习模型
- 支持评论时间排序和特征矩阵构建
- 谣言概率计算和风险等级评估
- 批量检测和结果导出

### 3. 人工校验模块
- 未处理/已处理文本分类管理
- 支持谣言/非谣言标记
- 处理日志记录和修改功能
- 批量校验操作

### 4. 关键词搜索模块
- 词云可视化展示热点关键词
- 关键词筛选和概率过滤
- 搜索建议和热门关键词
- 结果导出功能

## 技术栈

### 后端
- **框架**: Flask 2.3.3
- **数据库**: MySQL 8.0
- **缓存**: Redis 6.0
- **机器学习**: TensorFlow 2.13.0, scikit-learn 1.3.0
- **文本处理**: jieba 0.42.1
- **爬虫**: Scrapy 2.10.0

### 前端
- **框架**: Vue.js 2.6.14
- **UI组件**: Element UI 2.15.9
- **状态管理**: Vuex 3.6.2
- **路由**: Vue Router 3.5.1
- **HTTP客户端**: Axios 0.27.2

## 安装部署

### 1. 环境要求
- Python 3.8+
- Node.js 14+
- MySQL 8.0+
- Redis 6.0+

### 2. 后端部署

```bash
# 克隆项目
git clone <repository-url>
cd social_analysis

# 安装Python依赖
pip install -r requirements.txt

# 配置数据库
# 修改 app.py 中的数据库连接信息
# 创建数据库
mysql -u root -p < database/init.sql

# 启动Redis服务
redis-server

# 启动后端服务
python run.py
```

### 3. 前端部署

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run serve

# 构建生产版本
npm run build
```

## 使用说明

### 1. 数据展示
- 访问 `/data` 查看社交媒体数据
- 支持滚动加载和详情查看
- 显示数据统计信息

### 2. 检测结果
- 访问 `/detection` 查看检测结果
- 支持手动刷新和自动刷新
- 可进行批量检测和结果导出

### 3. 人工校验
- 访问 `/verification` 进行人工校验
- 支持未处理和已处理数据管理
- 可修改校验结果

### 4. 关键词搜索
- 访问 `/search` 进行关键词搜索
- 支持词云展示和关键词筛选
- 可导出搜索结果

## API接口

### 数据接口
- `GET /api/data` - 获取数据列表
- `GET /api/data/{id}` - 获取数据详情
- `POST /api/data/{id}/replies` - 添加回复

### 检测接口
- `POST /api/predict_rumor` - 预测谣言概率
- `POST /api/batch_predict` - 批量预测
- `POST /api/train_model` - 训练模型
- `GET /api/model_status` - 获取模型状态

### 校验接口
- `POST /api/verify-rumor` - 人工校验
- `POST /api/modify-verification` - 修改校验结果
- `GET /api/processed-data` - 获取已处理数据
- `GET /api/unprocessed-data` - 获取未处理数据

### 搜索接口
- `GET /api/wordcloud` - 生成词云
- `POST /api/filter-by-keyword` - 关键词筛选
- `GET /api/hot-keywords` - 获取热门关键词

## 数据库设计

### 主要表结构
- `rumor_data` - 谣言数据表
- `replies` - 回复数据表
- `process_log` - 处理日志表

### 索引优化
- 时间戳索引
- 状态索引
- 概率索引
- 来源平台索引

## 模型说明

### 特征提取
- 使用TF-IDF算法提取文本特征
- 支持中文分词和停用词过滤
- 构建特征向量矩阵

### 模型架构
- 嵌入层 (Embedding)
- LSTM层 (双向)
- 全连接层 (Dense)
- 输出层 (Sigmoid)

### 训练流程
1. 文本预处理和分词
2. 特征向量化
3. 序列填充和排序
4. 模型训练和验证
5. 模型保存和加载

## 配置说明

### 数据库配置
```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://username:password@localhost/rumor_db'
```

### Redis配置
```python
redis_client = redis.Redis(host='localhost', port=6379, db=0)
```

### 模型配置
```python
max_features = 5000
max_sequence_length = 200
```

## 开发指南

### 添加新功能
1. 在后端添加新的路由和视图函数
2. 在前端添加对应的Vue组件
3. 更新数据库模型（如需要）
4. 添加相应的测试用例

### 代码规范
- 后端遵循PEP 8规范
- 前端使用ESLint检查
- 数据库使用下划线命名
- API使用RESTful风格

## 常见问题

### 1. 数据库连接失败
- 检查MySQL服务是否启动
- 确认数据库连接信息正确
- 检查防火墙设置

### 2. Redis连接失败
- 检查Redis服务是否启动
- 确认端口6379是否开放
- 检查Redis配置

### 3. 模型加载失败
- 确认模型文件存在
- 检查TensorFlow版本兼容性
- 查看错误日志

### 4. 前端构建失败
- 检查Node.js版本
- 清除node_modules重新安装
- 检查网络连接

## 许可证

MIT License

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 联系方式

- 项目维护者: 未来媒体研究中心
- 邮箱: contact@example.com
- 项目地址: https://github.com/example/rumor-detection

