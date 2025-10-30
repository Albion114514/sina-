# 微博热搜谣言检测项目

## 项目简介
本项目用于实现对微博热搜关键词的相关文章进行谣言检测，通过关键词提取和谣言识别，帮助用户快速辨别微博热点内容的真实性。


## 功能说明
1. **微博数据爬取**：通过配置cookie获取微博热搜及相关文章数据（需在`scrap_data.py`中配置）。
2. **关键词提取**：针对微博热搜短标题，基于`jieba`库使用TextRank和TF-IDF算法提取关键信息（实现于`backend/init_data.py`）。
3. **谣言检测示例**：提供谣言检测数据结构示例，包含内容文本、评论及谣言标签（1表示谣言，0表示非谣言），见`backend/rumor_detect_model.py`。
4. **服务部署**：通过Node.js服务（`server.js`）提供后端接口支持，依赖Express框架实现HTTP服务及跨域处理。


## 环境要求
- Node.js（推荐v18及以上，兼容`express@5.1.0`要求）
- Python（用于运行关键词提取及数据处理脚本，推荐3.7+）
- 依赖库：
  - Node.js：`express`、`cors`（见`package.json`）
  - Python：`jieba`（用于中文分词及关键词提取）


## 安装步骤
1. 克隆或下载项目到本地
2. 进入项目主目录`rumors_detection`：
   ```bash
   cd 集合前端后轻量代码/rumors_detection
   ```
3. 安装Node.js依赖：
   ```bash
   npm install
   ```
4. 安装Python依赖：
   ```bash
   pip install jieba
   ```


## 运行方法
1. **配置Cookie**：  
   打开`scrap_data.py`，在`Scraper`类的初始化函数中，修改`weibo.com`和`weibo.cn`对应的客户端及移动端cookie（需自行获取有效cookie）。

2. **启动服务**：  
   方法一：直接运行脚本（Windows系统）：
   ```bash
   start.bat
   ```
   
   方法二：手动启动：
   - 激活Python虚拟环境（若使用）
   - 进入`rumors_detection`目录
   - 运行Node服务：
     ```bash
     node server.js
     ```


## 项目结构
```
集合前端后轻量代码/
├── rumors_detection/          # 主项目目录
│   ├── backend/               # 后端处理脚本
│   │   ├── init_data.py       # 微博热搜标题关键词提取
│   │   └── rumor_detect_model.py  # 谣言检测示例数据
│   ├── scrap_data.py          # 微博数据爬取（需配置cookie）
│   ├── server.js              # Node.js服务入口
│   ├── package.json           # Node.js依赖配置
│   ├── package-lock.json      # 依赖版本锁定
│   ├── start.bat              # 快捷启动脚本
│   └── README.txt             # 基础运行说明
└── social_analysis/           # 相关分析模块
    └── LICENSE                # Apache License 2.0许可证
```


## 许可证
本项目遵循Apache License 2.0开源协议，详情见`social_analysis/LICENSE`文件。
