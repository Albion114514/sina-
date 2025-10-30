# 社交媒体文本分析系统 - 前端程序

本目录包含社交媒体文本分析系统的前端实现代码，基于HTML、CSS和JavaScript开发，无需额外的构建工具即可运行。

## 项目结构

```
frontend/
├── index.html       # 主页面
├── css/
│   └── style.css    # 样式文件
├── js/
│   └── app.js       # JavaScript代码
└── images/
    └── default_wordcloud.txt  # 图片占位符
```

## 功能模块

前端程序包含三个主要功能模块：

### 1. 搜索模块
- 关键词搜索：输入关键词搜索相关的社交媒体文本内容
- 词云展示：展示搜索结果中的关键词词云
- 搜索结果分页展示：支持分页浏览搜索结果

### 2. 人工校验模块
- 查看待校验的文本内容
- 根据模型预测结果进行人工校验（确认为谣言或非谣言）
- 支持按关键词和状态过滤校验内容

### 3. 谣言检测模块
- 输入文本内容进行实时谣言检测
- 展示检测结果，包括谣言概率、风险等级和可信度

## 如何运行

1. 确保后端服务已经启动，默认监听在 `http://localhost:5000`
2. 在浏览器中直接打开 `frontend/index.html` 文件
3. 或者使用简单的HTTP服务器来运行：
   - 使用Python：在frontend目录下运行 `python -m http.server 8000`，然后访问 `http://localhost:8000`
   - 使用Node.js：在frontend目录下运行 `npx http-server -p 8000`，然后访问 `http://localhost:8000`

## 开发环境说明

- 前端使用原生HTML、CSS和JavaScript开发，无需复杂的构建流程
- 为了方便开发和测试，代码中包含了模拟数据功能，在本地环境下会自动启用
- 模拟数据包括：搜索结果、词云数据、待校验数据和检测结果

## 与后端API的交互

前端程序通过以下API与后端交互：

1. 搜索相关API：
   - `GET /api/search/trending` - 获取热门话题词云
   - `GET /api/search?keyword={keyword}&page={page}` - 搜索关键词
   - `GET /api/search/wordcloud?keyword={keyword}` - 获取关键词词云

2. 校验相关API：
   - `GET /api/verification/unverified` - 获取未校验数据
   - `GET /api/verification?keyword={keyword}&status={status}&page={page}` - 过滤校验数据
   - `POST /api/verification/confirm` - 确认校验结果

3. 检测相关API：
   - `POST /api/detection` - 进行谣言检测

## 注意事项

1. 模拟数据功能仅在本地环境（localhost或127.0.0.1）下启用
2. 当后端服务不可用时，前端会使用模拟数据展示界面，但实际功能会有限制
3. 为了获得最佳体验，请确保后端服务正常运行
4. 图片资源目前使用占位符，实际使用时需要替换为真实的图片文件

## 浏览器兼容性

支持现代浏览器的最新版本，包括：
- Google Chrome
- Mozilla Firefox
- Apple Safari
- Microsoft Edge

## 许可证

保留所有权利。