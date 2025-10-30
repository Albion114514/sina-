// 全局变量和初始化
const API_BASE_URL = 'http://localhost:5000';
let currentPage = 1;
let currentKeyword = '';
let currentStatus = '';

// 初始化函数
function init() {
    // 绑定事件处理函数
    bindEvents();
    // 初始化页面，默认显示搜索模块
    showTab('search');
}
// 补充CSV数据处理工具函数
document.addEventListener('DOMContentLoaded', () => {
    // 扩展Vue实例方法（如果需要）
    const app = Vue.app;
    if (app) {
        // 格式化日期显示
        app.config.globalProperties.$formatDate = (dateString) => {
            const date = new Date(dateString);
            return date.toLocaleString('zh-CN', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
            });
        };

        // 关键字高亮处理
        app.config.globalProperties.$highlightKeyword = (text, keyword) => {
            if (!keyword) return text;
            const regex = new RegExp(`(${keyword})`, 'gi');
            return text.replace(regex, '<span class="highlight">$1</span>');
        };
    }
});

// 样式补充（可迁移到style.css）
const style = document.createElement('style');
style.textContent = `
    .csv-upload {
        margin: 15px 0;
        padding: 10px;
        border-bottom: 1px solid #eee;
    }
    .upload-status {
        margin-left: 10px;
        color: #67c23a;
    }
    .highlight {
        color: #f56c6c;
        font-weight: bold;
    }
    .probability-display {
        margin: 10px 0;
        padding: 8px;
        background-color: #f5f7fa;
        border-radius: 4px;
    }
`;
document.head.appendChild(style);
// 绑定事件处理函数
function bindEvents() {
    // 导航栏切换事件
    document.querySelectorAll('.navbar .menu a').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const tab = this.getAttribute('data-tab');
            showTab(tab);
        });
    });

    // 搜索模块事件绑定
    document.getElementById('search-keyword-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchKeywords();
        }
    });
    document.getElementById('search-keyword-btn').addEventListener('click', searchKeywords);
    document.getElementById('clear-search-btn').addEventListener('click', clearSearch);

    // 校验模块事件绑定
    document.getElementById('verify-keyword-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            filterVerifications();
        }
    });
    document.getElementById('verify-filter-btn').addEventListener('click', filterVerifications);
    document.getElementById('verify-refresh-btn').addEventListener('click', loadUnverifiedData);

    // 检测模块事件绑定
    document.getElementById('detection-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            detectRumor();
        }
    });
    document.getElementById('detect-btn').addEventListener('click', detectRumor);

    // 导航栏切换按钮样式
    document.querySelectorAll('.navbar .menu a').forEach(link => {
        link.addEventListener('click', function() {
            document.querySelectorAll('.navbar .menu a').forEach(item => {
                item.classList.remove('active');
            });
            this.classList.add('active');
        });
    });
}

// 显示指定的标签页
function showTab(tabName) {
    // 隐藏所有标签页内容
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.add('hidden');
    });
    
    // 显示选中的标签页内容
    const selectedTab = document.getElementById(`${tabName}-content`);
    if (selectedTab) {
        selectedTab.classList.remove('hidden');
    }
    
    // 根据选中的标签页加载对应数据
    if (tabName === 'search') {
        loadInitialSearchData();
    } else if (tabName === 'verify') {
        loadUnverifiedData();
    }
}

// 初始化加载搜索数据
function loadInitialSearchData() {
    // 加载默认的热门话题词云
    fetch(`${API_BASE_URL}/api/search/trending`) 
        .then(response => response.json())
        .then(data => {
            if (data && data.wordcloud_path) {
                const wordcloudImg = document.getElementById('wordcloud-image');
                wordcloudImg.src = data.wordcloud_path;
                wordcloudImg.onclick = function() {
                    const word = prompt('请输入要搜索的关键词：');
                    if (word) {
                        document.getElementById('search-keyword-input').value = word;
                        searchKeywords();
                    }
                };
            }
        })
        .catch(error => {
            console.error('加载热门话题失败:', error);
            showError('加载热门话题失败');
        });
}

// 搜索关键词
function searchKeywords() {
    const keyword = document.getElementById('search-keyword-input').value.trim();
    if (!keyword) {
        showError('请输入关键词');
        return;
    }
    
    currentKeyword = keyword;
    currentPage = 1;
    
    showLoading('search-results');
    
    fetch(`${API_BASE_URL}/api/search?keyword=${encodeURIComponent(keyword)}&page=${currentPage}`)
        .then(response => response.json())
        .then(data => {
            renderSearchResults(data);
            renderWordcloud(keyword);
        })
        .catch(error => {
            console.error('搜索失败:', error);
            showError('搜索失败，请重试');
            hideLoading('search-results');
        });
}

// 渲染搜索结果
function renderSearchResults(data) {
    const resultsContainer = document.getElementById('search-results');
    resultsContainer.innerHTML = '';
    
    if (!data || !data.items || data.items.length === 0) {
        resultsContainer.innerHTML = '<div class="empty">暂无相关结果</div>';
        return;
    }
    
    data.items.forEach(item => {
        const contentItem = document.createElement('div');
        contentItem.className = 'content-item';
        
        let itemHtml = `
            <p><strong>内容：</strong>${item.content || '无内容'}</p>
            <div class="item-meta">
                <span>来源：${item.source || '未知'}</span>
                <span>发布时间：${item.publish_time || '未知'}</span>
                <span>热度：${item.popularity || 0}</span>
            </div>
        `;
        
        if (item.images && item.images.length > 0) {
            itemHtml = `<img src="${item.images[0]}" alt="内容图片">` + itemHtml;
        }
        
        contentItem.innerHTML = itemHtml;
        resultsContainer.appendChild(contentItem);
    });
    
    renderPagination(data.total, data.page, data.page_size);
    hideLoading('search-results');
}

// 渲染词云
function renderWordcloud(keyword) {
    fetch(`${API_BASE_URL}/api/search/wordcloud?keyword=${encodeURIComponent(keyword)}`)
        .then(response => response.json())
        .then(data => {
            if (data && data.wordcloud_path) {
                const wordcloudImg = document.getElementById('wordcloud-image');
                wordcloudImg.src = data.wordcloud_path;
                // 添加随机参数避免缓存
                wordcloudImg.src += '?' + new Date().getTime();
            }
        })
        .catch(error => {
            console.error('渲染词云失败:', error);
        });
}

// 渲染分页
function renderPagination(total, page, pageSize) {
    const totalPages = Math.ceil(total / pageSize);
    const paginationContainer = document.getElementById('pagination');
    paginationContainer.innerHTML = '';
    
    if (totalPages <= 1) return;
    
    let paginationHtml = `
        <button onclick="changePage(${page - 1})" ${page <= 1 ? 'disabled' : ''}>上一页</button>
        <span>第 ${page} 页 / 共 ${totalPages} 页</span>
        <button onclick="changePage(${page + 1})" ${page >= totalPages ? 'disabled' : ''}>下一页</button>
    `;
    
    paginationContainer.innerHTML = paginationHtml;
}

// 切换页码
function changePage(page) {
    if (page < 1) return;
    
    currentPage = page;
    
    showLoading('search-results');
    
    fetch(`${API_BASE_URL}/api/search?keyword=${encodeURIComponent(currentKeyword)}&page=${currentPage}`)
        .then(response => response.json())
        .then(data => {
            renderSearchResults(data);
        })
        .catch(error => {
            console.error('加载分页数据失败:', error);
            showError('加载失败，请重试');
            hideLoading('search-results');
        });
}

// 清除搜索
function clearSearch() {
    document.getElementById('search-keyword-input').value = '';
    document.getElementById('search-results').innerHTML = '';
    document.getElementById('pagination').innerHTML = '';
    loadInitialSearchData();
}

// 加载未校验数据
function loadUnverifiedData() {
    showLoading('verification-results');
    
    fetch(`${API_BASE_URL}/api/verification/unverified`)
        .then(response => response.json())
        .then(data => {
            renderVerificationResults(data);
        })
        .catch(error => {
            console.error('加载未校验数据失败:', error);
            showError('加载失败，请重试');
            hideLoading('verification-results');
        });
}

// 过滤校验数据
function filterVerifications() {
    const keyword = document.getElementById('verify-keyword-input').value.trim();
    const status = document.getElementById('verify-status-select').value;
    
    currentKeyword = keyword;
    currentStatus = status;
    currentPage = 1;
    
    showLoading('verification-results');
    
    let url = `${API_BASE_URL}/api/verification?`;
    if (keyword) url += `keyword=${encodeURIComponent(keyword)}&`;
    if (status) url += `status=${status}&`;
    url += `page=${currentPage}`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            renderVerificationResults(data);
        })
        .catch(error => {
            console.error('过滤校验数据失败:', error);
            showError('加载失败，请重试');
            hideLoading('verification-results');
        });
}

// 渲染校验结果
function renderVerificationResults(data) {
    const resultsContainer = document.getElementById('verification-results');
    resultsContainer.innerHTML = '';
    
    if (!data || !data.items || data.items.length === 0) {
        resultsContainer.innerHTML = '<div class="empty">暂无需要校验的数据</div>';
        return;
    }
    
    data.items.forEach(item => {
        const contentItem = document.createElement('div');
        contentItem.className = 'content-item';
        
        // 根据概率设置风险等级样式
        let riskLevelClass = 'low';
        let riskLevelText = '低风险';
        if (item.probability > 0.7) {
            riskLevelClass = 'high';
            riskLevelText = '高风险';
        } else if (item.probability > 0.3) {
            riskLevelClass = 'medium';
            riskLevelText = '中风险';
        }
        
        contentItem.innerHTML = `
            <p><strong>内容：</strong>${item.content || '无内容'}</p>
            <p><strong>检测概率：</strong><span class="probability ${riskLevelClass}">${riskLevelText} (${(item.probability * 100).toFixed(2)}%)</span></p>
            <p><strong>来源：</strong>${item.source || '未知'}</p>
            <p><strong>发布时间：</strong>${item.publish_time || '未知'}</p>
            <p><strong>当前状态：</strong>${getStatusText(item.status)}</p>
            <div class="item-meta">
                <button onclick="showVerifyDialog(${item.id}, '${escapeHTML(item.content)}', ${item.probability})" class="el-button">校验</button>
            </div>
        `;
        
        resultsContainer.appendChild(contentItem);
    });
    
    renderPagination(data.total, data.page, data.page_size);
    hideLoading('verification-results');
}

// 显示校验对话框
function showVerifyDialog(id, content, probability) {
    // 创建对话框
    const dialog = document.createElement('div');
    dialog.className = 'el-dialog';
    dialog.innerHTML = `
        <div class="el-dialog__wrapper">
            <div class="el-dialog">
                <div class="el-dialog__header">
                    <span class="el-dialog__title">人工校验</span>
                    <button type="button" class="el-dialog__headerbtn" onclick="document.body.removeChild(dialog)">
                        <i class="el-icon el-icon-close"></i>
                    </button>
                </div>
                <div class="el-dialog__body dialog-content">
                    <p><strong>内容：</strong>${content}</p>
                    <div class="probability-display">
                        <p><strong>模型预测：</strong>${getProbabilityText(probability)}</p>
                    </div>
                    <div class="confirm-text">请确认内容是否为谣言：</div>
                    <div class="dialog-buttons">
                        <button onclick="confirmVerification(${id}, 'true', dialog)" class="el-button el-button--danger">确认为谣言</button>
                        <button onclick="confirmVerification(${id}, 'false', dialog)" class="el-button">确认为非谣言</button>
                        <button onclick="document.body.removeChild(dialog)" class="el-button el-button--default">取消</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(dialog);
}

// 确认校验结果
function confirmVerification(id, isRumor, dialog) {
    fetch(`${API_BASE_URL}/api/verification/confirm`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            id: id,
            is_rumor: isRumor === 'true'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showSuccess('校验成功');
            document.body.removeChild(dialog);
            // 重新加载数据
            if (currentStatus === 'unverified') {
                loadUnverifiedData();
            } else {
                filterVerifications();
            }
        } else {
            showError('校验失败：' + (data.message || '未知错误'));
        }
    })
    .catch(error => {
        console.error('校验失败:', error);
        showError('校验失败，请重试');
    });
}

// 谣言检测
function detectRumor() {
    const content = document.getElementById('detection-input').value.trim();
    if (!content) {
        showError('请输入要检测的内容');
        return;
    }
    
    showLoading('detection-results');
    
    fetch(`${API_BASE_URL}/api/detection`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ content: content })
    })
    .then(response => response.json())
    .then(data => {
        renderDetectionResults(data);
    })
    .catch(error => {
        console.error('检测失败:', error);
        showError('检测失败，请重试');
        hideLoading('detection-results');
    });
}

// 渲染检测结果
function renderDetectionResults(data) {
    const resultsContainer = document.getElementById('detection-results');
    
    if (!data || !data.success) {
        resultsContainer.innerHTML = '<div class="empty">检测失败，请重试</div>';
        return;
    }
    
    // 根据概率设置风险等级
    let riskLevelClass = 'low';
    let riskLevelText = '低风险';
    if (data.probability > 0.7) {
        riskLevelClass = 'high';
        riskLevelText = '高风险';
    } else if (data.probability > 0.3) {
        riskLevelClass = 'medium';
        riskLevelText = '中风险';
    }
    
    resultsContainer.innerHTML = `
        <div class="result-card">
            <div class="result-item">
                <div class="label">检测内容：</div>
                <div class="value">${data.content || '无内容'}</div>
            </div>
            <div class="result-item">
                <div class="label">检测结果：</div>
                <div class="value">
                    <span class="probability ${riskLevelClass}">${riskLevelText}</span>
                </div>
            </div>
            <div class="result-item">
                <div class="label">谣言概率：</div>
                <div class="value">${(data.probability * 100).toFixed(2)}%</div>
            </div>
            <div class="result-item">
                <div class="label">可信度：</div>
                <div class="value">${getConfidenceText(data.confidence)}</div>
            </div>
            <div class="result-item">
                <div class="label">检测时间：</div>
                <div class="value">${new Date().toLocaleString()}</div>
            </div>
        </div>
    `;
    
    hideLoading('detection-results');
}

// 辅助函数
function getStatusText(status) {
    switch(status) {
        case 'unverified': return '待校验';
        case 'rumor': return '已确认为谣言';
        case 'not_rumor': return '已确认为非谣言';
        default: return '未知';
    }
}

function getProbabilityText(probability) {
    if (probability > 0.7) {
        return `高风险谣言 (${(probability * 100).toFixed(2)}%)`;
    } else if (probability > 0.3) {
        return `中等风险 (${(probability * 100).toFixed(2)}%)`;
    } else {
        return `低风险 (${(probability * 100).toFixed(2)}%)`;
    }
}

function getConfidenceText(confidence) {
    if (!confidence) return '未知';
    if (confidence > 0.8) return '高';
    if (confidence > 0.5) return '中';
    return '低';
}

function showLoading(containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '<div class="loading">加载中...</div>';
}

function hideLoading(containerId) {
    // 加载完成后会被实际内容替换，这里不需要额外操作
}

function showError(message) {
    // 简单的错误提示，实际项目中可以使用更友好的提示组件
    alert(`错误：${message}`);
}

function showSuccess(message) {
    // 简单的成功提示，实际项目中可以使用更友好的提示组件
    alert(`成功：${message}`);
}

function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// 页面加载完成后初始化
window.addEventListener('DOMContentLoaded', init);

// 模拟API数据（用于开发和测试）
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    // 在开发环境下，启用模拟数据功能
    const originalFetch = window.fetch;
    window.fetch = function(url, options) {
        // 模拟API响应
        if (url.includes('/api/search/trending')) {
            return Promise.resolve({
                json: () => Promise.resolve({
                    wordcloud_path: '../images/default_wordcloud.png'
                })
            });
        }
        
        if (url.includes('/api/search?keyword=')) {
            const keyword = url.match(/keyword=([^&]+)/)[1];
            return Promise.resolve({
                json: () => Promise.resolve({
                    items: [
                        {
                            content: `这是关于${decodeURIComponent(keyword)}的第一条搜索结果内容。这是一个示例数据，用于前端开发测试。`,
                            source: '微博',
                            publish_time: '2023-10-27 10:30:00',
                            popularity: 12580,
                            images: []
                        },
                        {
                            content: `这是关于${decodeURIComponent(keyword)}的第二条搜索结果内容。包含了一些相关的讨论和观点。`,
                            source: '知乎',
                            publish_time: '2023-10-27 09:15:00',
                            popularity: 8920
                        }
                    ],
                    total: 2,
                    page: 1,
                    page_size: 10
                })
            });
        }
        
        if (url.includes('/api/search/wordcloud')) {
            return Promise.resolve({
                json: () => Promise.resolve({
                    wordcloud_path: '../images/default_wordcloud.png'
                })
            });
        }
        
        if (url.includes('/api/verification/unverified') || url.includes('/api/verification?')) {
            return Promise.resolve({
                json: () => Promise.resolve({
                    items: [
                        {
                            id: 1,
                            content: '这是一条可能的谣言内容，需要人工校验。系统检测到有较高的谣言概率。',
                            probability: 0.85,
                            source: '微博',
                            publish_time: '2023-10-27 08:00:00',
                            status: 'unverified'
                        },
                        {
                            id: 2,
                            content: '这条信息的真实性存疑，需要进一步核实。系统给出的谣言概率为中等。',
                            probability: 0.45,
                            source: '微信朋友圈',
                            publish_time: '2023-10-27 07:30:00',
                            status: 'unverified'
                        },
                        {
                            id: 3,
                            content: '这是一条普通信息，系统检测到谣言概率较低，但仍需要人工确认。',
                            probability: 0.20,
                            source: '新闻网站',
                            publish_time: '2023-10-27 06:45:00',
                            status: 'unverified'
                        }
                    ],
                    total: 3,
                    page: 1,
                    page_size: 10
                })
            });
        }
        
        if (url.includes('/api/verification/confirm')) {
            return Promise.resolve({
                json: () => Promise.resolve({
                    success: true,
                    message: '校验成功'
                })
            });
        }
        
        if (url.includes('/api/detection')) {
            const content = JSON.parse(options.body).content;
            return Promise.resolve({
                json: () => Promise.resolve({
                    success: true,
                    content: content,
                    probability: Math.random() * 0.7 + 0.1, // 随机概率 0.1-0.8
                    confidence: Math.random() * 0.3 + 0.7  // 随机可信度 0.7-1.0
                })
            });
        }
        
        // 其他请求正常处理
        return originalFetch(url, options);
    };
}