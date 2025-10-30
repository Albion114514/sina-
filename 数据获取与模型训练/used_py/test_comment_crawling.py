# test_comments.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试移动端评论爬取功能
"""

import json
import sys
import os

# 添加当前目录到路径，以便导入移动端爬虫
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from weibo_mobile_crawler import WeiboMobileCrawler

    print("✅ 成功导入移动端爬虫")
except ImportError:
    print("❌ 无法导入移动端爬虫，请确保 weibo_mobile_crawler.py 存在")
    sys.exit(1)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_comment_crawling():
    """测试移动端评论爬取"""
    print("🧪 测试移动端评论爬取功能")
    print("=" * 50)

    # 加载配置
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"crawl_comments 配置: {config.get('crawl_comments', '未设置')}")

    # 创建移动端爬虫实例
    crawler = WeiboMobileCrawler()

    # 测试连接数据库
    if crawler.connect_database():
        print("✅ 数据库连接成功")
    else:
        print("❌ 数据库连接失败")
        return

    # 测试获取关键字ID
    keyword = config['keywords'][0]
    keyword_id = crawler.get_keyword_id(keyword)
    print(f"关键字 '{keyword}' 的ID: {keyword_id}")

    # 测试爬取一页数据（使用移动端API）
    print(f"\n测试爬取关键词: {keyword}")
    weibo_items = crawler.crawl_weibo_search_api(keyword, 1)
    print(f"找到 {len(weibo_items)} 条微博")

    # 显示第一条微博的详细信息
    if weibo_items:
        test_weibo = weibo_items[0]
        print(f"\n📝 第一条微博示例：")
        print(f"  ID: {test_weibo['id']}")
        print(f"  用户: {test_weibo.get('user_name', '未知')}")
        print(f"  内容: {test_weibo['text'][:100]}...")
        print(f"  转发: {test_weibo.get('reposts_count', 0)}")
        print(f"  评论: {test_weibo.get('comments_count', 0)}")
        print(f"  点赞: {test_weibo.get('attitudes_count', 0)}")

    # 测试评论爬取
    if weibo_items and config.get('crawl_comments', False):
        test_weibo = weibo_items[0]
        print(f"\n测试爬取微博 {test_weibo['id']} 的评论")
        comments = crawler.crawl_weibo_comments_api(test_weibo['id'])
        print(f"获取到 {len(comments)} 条评论")

        if comments:
            saved_count = crawler.save_comments_to_database(comments)
            print(f"成功保存 {saved_count} 条评论到数据库")

            # 显示前几条评论
            print("\n📝 评论示例：")
            for i, comment in enumerate(comments[:3]):
                print(f"  {i + 1}. {comment['user']}: {comment['content'][:50]}...")

    # 测试保存微博数据
    if weibo_items:
        print(f"\n💾 测试保存微博数据...")
        saved_count = 0
        for item in weibo_items[:2]:  # 只保存前两条测试
            if crawler.save_to_database(item, keyword_id):
                saved_count += 1
                print(f"  保存微博: {item['id']}")

        print(f"成功保存 {saved_count} 条微博到数据库")

    if crawler.db_connection:
        crawler.db_connection.close()
    print("\n✅ 移动端爬虫测试完成")


if __name__ == "__main__":
    test_comment_crawling()