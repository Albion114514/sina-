# weibo_crawler_fixed.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版微博爬虫 - 支持评论爬取和关键字关联
"""
import traceback
import json
import time
import re
import requests
import pymysql
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import quote
import logging
import sys
import os
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weibo_crawler_fixed.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class WeiboCrawlerFixed:
    def __init__(self, config_path='config.json'):
        """初始化爬虫"""
        self.config = self.load_config(config_path)
        self.session = requests.Session()
        self.setup_session()
        self.db_connection = None

    def load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"配置文件加载失败：{e}")
            raise

    def setup_session(self):
        """设置请求会话"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://weibo.com/',
        }

        if 'headers' in self.config:
            headers.update(self.config['headers'])

        self.session.headers.update(headers)

        if self.config.get('proxies'):
            self.session.proxies.update(self.config['proxies'])

    def get_random_delay(self):
        """获取随机延迟时间"""
        delay_config = self.config['crawler_settings']['request_delay']
        if isinstance(delay_config, (list, tuple)) and len(delay_config) == 2:
            return random.uniform(delay_config[0], delay_config[1])
        return delay_config if delay_config else 2

    def connect_database(self):
        """连接数据库"""
        try:
            mysql_config = self.config['mysql_config'].copy()
            mysql_config['charset'] = 'utf8mb4'

            # 确保端口正确
            if 'port' not in mysql_config:
                mysql_config['port'] = 3307  # 根据你的配置

            self.db_connection = pymysql.connect(**mysql_config)
            logger.info("数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败：{e}")
            return False

    def safe_get_text(self, element, default=''):
        """安全获取文本内容"""
        try:
            return element.get_text(strip=True) if element else default
        except Exception:
            return default

    def get_keyword_id(self, keyword):
        """获取关键字ID"""
        if not self.db_connection:
            return None

        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT id FROM keywords WHERE keyword = %s", (keyword,))
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"获取关键字ID错误：{e}")
            return None

    def parse_weibo_item(self, item):
        """解析单个微博条目 - 改进版本"""
        try:
            # 获取微博ID - 多种方式尝试
            weibo_id = ''

            # 方式1: 从mid属性获取
            mid_match = re.search(r'mid="(\d+)"', str(item))
            if mid_match:
                weibo_id = mid_match.group(1)

            # 方式2: 从数据模块获取
            if not weibo_id:
                module_match = re.search(r'&id=(\d+)', str(item))
                if module_match:
                    weibo_id = module_match.group(1)

            # 方式3: 从链接获取
            if not weibo_id:
                link_elem = item.find('a', href=re.compile(r'weibo\.com/\d+/(\w+)'))
                if link_elem:
                    href = link_elem.get('href', '')
                    weibo_match = re.search(r'weibo\.com/\d+/(\w+)', href)
                    if weibo_match:
                        weibo_id = weibo_match.group(1)

            # 方式4: 从评论链接获取
            if not weibo_id:
                comment_links = item.find_all('a', href=re.compile(r'comment'))
                for link in comment_links:
                    href = link.get('href', '')
                    id_match = re.search(r'&id=(\d+)', href)
                    if id_match:
                        weibo_id = id_match.group(1)
                        break

            # 如果还是没找到，使用时间戳（作为最后手段）
            if not weibo_id:
                weibo_id = str(int(time.time() * 1000))
                logger.warning(f"使用时间戳作为微博ID: {weibo_id}")

            # 获取微博内容
            content_elem = item.find('div', class_='content')
            if not content_elem:
                return None

            # 获取正文 - 多种选择器尝试
            text = ''
            text_selectors = [
                'p.txt', 'div.txt', 'p.content', 'div.content'
            ]

            for selector in text_selectors:
                text_elem = content_elem.select_one(selector)
                if text_elem:
                    text = self.safe_get_text(text_elem)
                    if text:
                        break

            # 清理文本
            if text:
                text = re.sub(r'\s+', ' ', text).strip()

            # 获取发布时间
            time_elem = content_elem.find('p', class_='from')
            time_str = ''
            if time_elem:
                time_link = time_elem.find('a')
                time_str = self.safe_get_text(time_link)

            timestamp = self.parse_time_string(time_str) if time_str else int(time.time())

            # 获取图片
            pics = []
            pic_elems = content_elem.find_all('img')
            for img in pic_elems:
                src = img.get('src', '')
                if src and 'sinaimg.cn' in src:
                    # 转换为高清图片URL
                    if 'orj360' in src:
                        src = src.replace('orj360', 'large')
                    elif 'thumbnail' in src:
                        src = src.replace('thumbnail', 'large')
                    pics.append(src)

            return {
                'id': weibo_id,
                'text': text,
                'pics': ','.join(pics),
                'timestamp': timestamp,
                'source': '新浪微博'
            }

        except Exception as e:
            logger.error(f"解析微博条目错误：{e}")
            return None

    def parse_time_string(self, time_str):
        """解析时间字符串"""
        try:
            now = datetime.now()

            if '分钟前' in time_str:
                minutes = int(re.search(r'(\d+)', time_str).group(1))
                return int((now - timedelta(minutes=minutes)).timestamp())
            elif '小时前' in time_str:
                hours = int(re.search(r'(\d+)', time_str).group(1))
                return int((now - timedelta(hours=hours)).timestamp())
            elif '今天' in time_str:
                time_part = re.search(r'今天\s*(\d+):(\d+)', time_str)
                if time_part:
                    hour, minute = int(time_part.group(1)), int(time_part.group(2))
                    return int(now.replace(hour=hour, minute=minute, second=0).timestamp())
            elif '月' in time_str and '日' in time_str:
                match = re.search(r'(\d+)月(\d+)日', time_str)
                if match:
                    month, day = int(match.group(1)), int(match.group(2))
                    year = now.year
                    return int(datetime(year, month, day).timestamp())

            return int(now.timestamp())

        except Exception as e:
            logger.error(f"时间解析错误：{e}")
            return int(time.time())

    def crawl_weibo_comments(self, weibo_id):
        """爬取单条微博的评论 - 改进版本"""
        if not weibo_id or weibo_id == '0' or len(weibo_id) < 10:
            return []

        try:
            # 使用不同的评论API
            url = "https://weibo.com/ajax/statuses/buildComments"
            params = {
                "is_reload": 1,
                "id": weibo_id,
                "is_show_bulletin": 2,
                "is_mix": 0,
                "count": 10,
                "uid": "",
                "fetch_level": 0
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': f'https://weibo.com/{weibo_id}',
                'X-Requested-With': 'XMLHttpRequest',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            }

            # 使用配置中的Cookie
            if 'headers' in self.config and 'Cookie' in self.config['headers']:
                headers['Cookie'] = self.config['headers']['Cookie']

            logger.info(f"爬取评论: {weibo_id}")

            # 增加超时和重试
            for attempt in range(3):
                try:
                    response = self.session.get(url, params=params, headers=headers, timeout=15)

                    if response.status_code == 200:
                        data = response.json()
                        break
                    elif response.status_code == 403:
                        logger.warning(f"评论请求被拒绝，状态码：{response.status_code}")
                        return []
                    else:
                        logger.warning(f"评论请求失败，状态码：{response.status_code}，尝试 {attempt + 1}/3")
                        time.sleep(2)
                except Exception as e:
                    logger.warning(f"评论请求异常，尝试 {attempt + 1}/3: {e}")
                    time.sleep(2)
            else:
                logger.error("评论请求多次失败")
                return []

            # 检查响应数据
            if not data:
                logger.warning("评论响应数据为空")
                return []

            # 不同的数据格式处理
            comments = []

            # 格式1: 直接包含data数组
            if 'data' in data and isinstance(data['data'], list):
                comment_list = data['data']
            # 格式2: 嵌套的data结构
            elif 'data' in data and 'data' in data['data']:
                comment_list = data['data']['data']
            else:
                logger.warning(f"评论数据格式未知: {data.keys() if data else '空数据'}")
                return []

            for item in comment_list:
                try:
                    comment = {
                        "id": str(item.get("id", "")),
                        "weibo_id": weibo_id,
                        "user": item.get("user", {}).get("screen_name", ""),
                        "content": item.get("text", "") or item.get("text_raw", ""),
                        "timestamp": item.get("created_at", "")
                    }

                    # 清理HTML标签
                    if comment["content"]:
                        comment["content"] = re.sub(r'<[^>]+>', '', comment["content"])

                    # 转换时间戳
                    try:
                        if comment["timestamp"]:
                            # 尝试多种时间格式
                            try:
                                dt = datetime.strptime(comment["timestamp"], "%a %b %d %H:%M:%S %z %Y")
                            except ValueError:
                                try:
                                    dt = datetime.strptime(comment["timestamp"], "%Y-%m-%d %H:%M:%S")
                                except ValueError:
                                    dt = datetime.now()
                            comment["timestamp"] = int(dt.timestamp())
                        else:
                            comment["timestamp"] = int(time.time())
                    except:
                        comment["timestamp"] = int(time.time())

                    # 只保存有内容的评论
                    if comment["content"].strip():
                        comments.append(comment)

                except Exception as e:
                    logger.warning(f"解析评论项失败: {e}")
                    continue

            logger.info(f"成功获取 {len(comments)} 条评论")
            return comments

        except Exception as e:
            logger.error(f"爬取评论错误：{e}")
            return []

    def save_to_database(self, weibo_data, keyword_id):
        """保存微博数据到数据库"""
        if not weibo_data or not self.db_connection or not keyword_id:
            return False

        try:
            with self.db_connection.cursor() as cursor:
                sql = """
                INSERT INTO weibo_data (id, keyword_id, text, pics, timestamp, source)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                text = VALUES(text),
                pics = VALUES(pics),
                timestamp = VALUES(timestamp),
                source = VALUES(source),
                updated_at = CURRENT_TIMESTAMP
                """

                cursor.execute(sql, (
                    weibo_data['id'],
                    keyword_id,
                    weibo_data['text'],
                    weibo_data['pics'],
                    weibo_data['timestamp'],
                    weibo_data['source']
                ))

                self.db_connection.commit()
                logger.info(f"保存微博成功：{weibo_data['id']}")
                return True

        except Exception as e:
            logger.error(f"数据库保存错误：{e}")
            if self.db_connection:
                self.db_connection.rollback()
            return False

    def save_comments_to_database(self, comments):
        """保存评论到数据库"""
        if not comments or not self.db_connection:
            return 0

        saved_count = 0
        try:
            with self.db_connection.cursor() as cursor:
                sql = """
                INSERT INTO comments (id, weibo_id, user, content, timestamp)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                user = VALUES(user),
                content = VALUES(content),
                timestamp = VALUES(timestamp),
                updated_at = CURRENT_TIMESTAMP
                """

                for comment in comments:
                    try:
                        cursor.execute(sql, (
                            comment['id'],
                            comment['weibo_id'],
                            comment['user'],
                            comment['content'],
                            comment['timestamp']
                        ))
                        saved_count += 1
                    except Exception as e:
                        logger.warning(f"保存评论 {comment.get('id')} 失败：{e}")
                        continue

                self.db_connection.commit()
                return saved_count

        except Exception as e:
            logger.error(f"评论批量保存错误：{e}")
            if self.db_connection:
                self.db_connection.rollback()
            return saved_count

    def crawl_weibo_search(self, keyword, page=1):
        """爬取微博搜索页面"""
        try:
            encoded_keyword = quote(keyword.encode('utf-8'))
            url = f"https://s.weibo.com/weibo?q={encoded_keyword}&page={page}"

            logger.info(f"爬取URL: {url}")

            response = self.session.get(url, timeout=20)

            if response.status_code != 200:
                logger.error(f"请求失败，状态码：{response.status_code}")
                return []

            # 编码处理
            try:
                response.encoding = 'utf-8'
                content = response.text
            except UnicodeDecodeError:
                try:
                    response.encoding = 'gb18030'
                    content = response.text
                except UnicodeDecodeError:
                    content = response.content.decode('utf-8', errors='ignore')

            # 检查反爬
            if any(pattern in content for pattern in ['异常请求', '验证码', 'security.weibo.com']):
                logger.warning("遇到反爬机制")
                return []

            # 解析HTML
            soup = BeautifulSoup(content, 'html.parser')
            weibo_items = []

            # 查找微博卡片
            selectors = ['div.card', 'div[action-type="feed_list_item"]', 'div.web-feed']

            for selector in selectors:
                card_elems = soup.select(selector)
                if card_elems:
                    for card in card_elems:
                        if card.find('div', class_=re.compile('content')):
                            weibo_data = self.parse_weibo_item(card)
                            if weibo_data and weibo_data['text']:
                                weibo_items.append(weibo_data)
                    break

            logger.info(f"找到 {len(weibo_items)} 条微博")
            return weibo_items

        except Exception as e:
            logger.error(f"爬取搜索页面错误：{e}")
            return []

    def crawl_keywords(self):
        """爬取所有关键词"""
        if not self.connect_database():
            return False

        try:
            keywords = self.config['keywords']
            start_page = self.config['startPage']
            max_page = self.config['maxPage']

            # 获取评论爬取配置并添加调试信息
            crawl_comments = self.config.get('crawl_comments', False)
            logger.info(f"评论爬取配置: crawl_comments = {crawl_comments}")
            logger.info(f"完整配置: {self.config.get('crawl_comments', '未找到')}")

            total_saved = 0
            total_comments_saved = 0

            for keyword in keywords:
                logger.info(f"开始爬取关键词：{keyword}")

                # 获取关键字ID
                keyword_id = self.get_keyword_id(keyword)
                if not keyword_id:
                    logger.error(f"无法获取关键字 '{keyword}' 的ID，跳过")
                    continue

                for page in range(start_page, max_page + 1):
                    logger.info(f"爬取第 {page} 页")

                    weibo_items = self.crawl_weibo_search(keyword, page)

                    saved_count = 0
                    for item in weibo_items:
                        if self.save_to_database(item, keyword_id):
                            saved_count += 1

                            # 只有在配置开启时才爬取评论
                            # 在 crawl_keywords 方法中简化评论爬取逻辑
                            if self.config.get('crawl_comments', False):
                                logger.info(f"评论爬取已启用，开始爬取微博 {item['id']} 的评论")
                                comments = self.crawl_weibo_comments(item['id'])
                                if comments:
                                    comments_saved = self.save_comments_to_database(comments)
                                    total_comments_saved += comments_saved
                                    logger.info(f"微博 {item['id']} 保存了 {comments_saved} 条评论")
                                else:
                                    logger.info(f"微博 {item['id']} 没有获取到评论")

                                # 评论爬取延迟
                                time.sleep(self.get_random_delay() / 2)
                            else:
                                logger.info(f"跳过评论爬取 - crawl_comments: {crawl_comments}, 微博ID: {item['id']}")

                    total_saved += saved_count
                    logger.info(f"第 {page} 页保存了 {saved_count} 条微博")

                    # 如果没有数据，停止翻页
                    if len(weibo_items) == 0:
                        logger.info(f"关键词 {keyword} 第 {page} 页无数据，停止爬取")
                        break

                    # 随机延迟
                    delay = self.get_random_delay()
                    logger.info(f"等待 {delay:.2f} 秒后继续")
                    time.sleep(delay)

                logger.info(f"关键词 {keyword} 爬取完成")

            logger.info(f"所有关键词爬取完成，共保存 {total_saved} 条微博，{total_comments_saved} 条评论")
            return True

        except Exception as e:
            logger.error(f"爬取过程出错：{e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            if self.db_connection:
                self.db_connection.close()
                logger.info("数据库连接已关闭")


# 修改 main 函数中的数据库初始化部分
def main():
    """主函数"""
    print("🚀 微博爬虫启动中...")
    print("=" * 50)

    try:
        # 先检查数据库表是否存在，如果不存在则创建
        try:
            from db_setup2_0 import create_tables, load_config
            config = load_config()
            mysql_config = config['mysql_config']
            keywords = config.get('keywords', [])
            create_tables(mysql_config, keywords)
            print("✅ 数据库表检查完成")
        except Exception as e:
            print(f"⚠️ 数据库表检查失败，但继续执行: {e}")

        crawler = WeiboCrawlerFixed()

        if crawler.crawl_keywords():
            print("✅ 爬取完成！")
        else:
            print("❌ 爬取失败")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断爬取")
    except Exception as e:
        print(f"❌ 程序运行错误：{e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()