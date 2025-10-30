
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weibo_mobile_crawler_main.py
移动端微博爬虫（基于 m.weibo.cn 接口），与项目现有表结构/配置兼容。

特性
- 读取 config.json：keywords、startPage、maxPage、crawler_settings、headers、proxies、mysql_config
- 抓取搜索结果（cards.card_type==9 的微博），解析文本/图片/时间/用户/计数
- 可选抓取热评接口（comments/hotflow）
- 数据库使用 pymysql，写入 weibo_data / comments（INSERT ... ON DUPLICATE KEY UPDATE）
- 关键词若未存在于 keywords 表，会自动插入再取 ID（满足外键约束）
- 日志清晰：关键词->分页->抓取条数->入库条数；错误/限流时明确提示
- 可选：如同目录存在 db_bootstrap.py，则以子进程方式执行 “--mode init” 做无损建表校验

兼容表结构
- weibo_data(id, keyword_id, text, pics, timestamp, source, user_name, reposts_count, comments_count, attitudes_count, created_at, updated_at)
- comments(id, weibo_id, user, content, timestamp, created_at, updated_at)
- keywords(id, keyword)
"""

import json
import os
import re
import sys
import time
import random
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import requests
import pymysql

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("weibo_mobile_crawler.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ---------------- Config -----------------
CONFIG_PATH = "config.json"

def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- Utils ------------------
def parse_api_time(time_str: str) -> int:
    """把 m.weibo.cn 返回的时间字符串转成秒级时间戳"""
    try:
        now = datetime.now()
        s = str(time_str)
        if "刚刚" in s:
            return int(now.timestamp())
        m = re.search(r"(\d+)\s*分钟前", s)
        if m:
            return int((now - timedelta(minutes=int(m.group(1)))).timestamp())
        h = re.search(r"(\d+)\s*小时前", s)
        if h:
            return int((now - timedelta(hours=int(h.group(1)))).timestamp())
        y = re.search(r"昨天\s*(\d{1,2}):(\d{1,2})", s)
        if y:
            hour, minute = int(y.group(1)), int(y.group(2))
            yest = now - timedelta(days=1)
            return int(yest.replace(hour=hour, minute=minute, second=0, microsecond=0).timestamp())

        # 常见格式兜底
        for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S"):
            try:
                return int(datetime.strptime(s, fmt).timestamp())
            except Exception:
                pass
        return int(now.timestamp())
    except Exception:
        return int(time.time())

def get_delay(delay_cfg) -> float:
    if isinstance(delay_cfg, (list, tuple)) and len(delay_cfg) == 2:
        return random.uniform(delay_cfg[0], delay_cfg[1])
    try:
        return float(delay_cfg)
    except Exception:
        return 2.0

def ensure_db(cfg):
    if Path("db_bootstrap.py").exists():
        try:
            logger.info("检测到 db_bootstrap.py，执行无损建表 (--mode init)")
            env = os.environ.copy()
            # 强制子进程用 UTF-8 输出，避免 Windows 控制台 GBK 误解码
            env["PYTHONIOENCODING"] = "utf-8"
            subprocess.run(
                [sys.executable, "db_bootstrap.py", "--mode", "init"],
                check=False,
                capture_output=True,   # 如需看到子进程原样输出，可改为 False
                text=True,
                encoding="utf-8",      # 关键：按 UTF-8 解码 stdout/stderr
                errors="replace",      # 或 "ignore"：遇到不可解码字符不报错
                env=env,
            )
        except Exception as e:
            logger.warning("执行 db_bootstrap.py 失败（忽略继续）：%s", e)

# ---------------- DB ---------------------
class DB:
    def __init__(self, mysql_cfg: Dict[str, Any]):
        cfg = mysql_cfg.copy()
        cfg.setdefault("charset", "utf8mb4")
        cfg["port"] = int(cfg.get("port", 3306))
        self.conn = pymysql.connect(**cfg)

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def ensure_keyword_id(self, keyword: str) -> Optional[int]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM keywords WHERE keyword=%s", (keyword,))
            row = cur.fetchone()
            if row:
                return int(row[0])
            # 插入再取
            cur.execute("INSERT INTO keywords (keyword) VALUES (%s)", (keyword,))
            kid = int(cur.lastrowid)
        self.conn.commit()
        return kid

    def save_weibo(self, item: Dict[str, Any], keyword_id: int) -> bool:
        sql = """
        INSERT INTO weibo_data (
            id, keyword_id, text, pics, timestamp, source,
            user_name, reposts_count, comments_count, attitudes_count
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            text=VALUES(text),
            pics=VALUES(pics),
            timestamp=VALUES(timestamp),
            source=VALUES(source),
            user_name=VALUES(user_name),
            reposts_count=VALUES(reposts_count),
            comments_count=VALUES(comments_count),
            attitudes_count=VALUES(attitudes_count),
            updated_at=CURRENT_TIMESTAMP
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (
                    item["id"],
                    keyword_id,
                    item.get("text", ""),
                    item.get("pics", ""),
                    int(item.get("timestamp", int(time.time()))),
                    item.get("source", "新浪微博移动端"),
                    item.get("user_name", ""),
                    int(item.get("reposts_count", 0)),
                    int(item.get("comments_count", 0)),
                    int(item.get("attitudes_count", 0)),
                ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error("保存微博失败 id=%s：%s", item.get("id"), e)
            self.conn.rollback()
            return False

    def save_comments(self, comments: List[Dict[str, Any]]) -> int:
        if not comments:
            return 0
        sql = """
        INSERT INTO comments (id, weibo_id, user, content, timestamp)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            user=VALUES(user),
            content=VALUES(content),
            timestamp=VALUES(timestamp),
            updated_at=CURRENT_TIMESTAMP
        """
        n = 0
        try:
            with self.conn.cursor() as cur:
                for c in comments:
                    try:
                        cur.execute(sql, (
                            c["id"],
                            c["weibo_id"],
                            c.get("user", ""),
                            c.get("content", ""),
                            int(c.get("timestamp", int(time.time()))),
                        ))
                        n += 1
                    except Exception as e:
                        logger.debug("忽略单条评论保存错误 id=%s：%s", c.get("id"), e)
            self.conn.commit()
            return n
        except Exception as e:
            logger.error("批量保存评论失败：%s", e)
            self.conn.rollback()
            return n

# --------------- Crawler -----------------
class WeiboMobileCrawler:
    BASE = "https://m.weibo.cn"

    def __init__(self, config_path: str = CONFIG_PATH):
        self.cfg = load_config(config_path)
        self.session = requests.Session()
        self._setup_session()
        self.db = DB(self.cfg["mysql_config"])

        cs = self.cfg.get("crawler_settings", {})
        self.delay_cfg = cs.get("request_delay", [10, 15])
        self.timeout = int(cs.get("timeout", 20))

        self.crawl_comments = bool(self.cfg.get("crawl_comments", False))
        self.max_comment_pages = int(self.cfg.get("max_comment_pages", 1))

    def _setup_session(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://m.weibo.cn/",
            "X-Requested-With": "XMLHttpRequest",
            "Connection": "keep-alive",
        }
        headers.update(self.cfg.get("headers", {}))
        self.session.headers.update(headers)

        proxies = self.cfg.get("proxies") or {}
        if proxies:
            self.session.proxies.update(proxies)

    # ---------- API calls ----------
    def _search_page(self, keyword: str, page: int) -> List[Dict[str, Any]]:
        """请求搜索页，返回标准化的微博条目列表"""
        url = f"{self.BASE}/api/container/getIndex"
        params = {
            "containerid": f"100103type=1&q={keyword}",
            "page_type": "searchall",
            "page": page,
        }
        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.status_code != 200:
            logger.warning("搜索接口状态码异常 %s：%s", r.status_code, r.text[:200])
            return []

        data = r.json()
        ok = data.get("ok")
        if ok != 1:
            logger.warning("搜索接口返回错误 ok=%s msg=%s", ok, data.get("msg"))
            return []

        cards = (data.get("data") or {}).get("cards") or []
        items: List[Dict[str, Any]] = []
        for card in cards:
            if card.get("card_type") != 9:
                continue
            mblog = card.get("mblog") or {}
            wid = mblog.get("id")
            if not wid:
                continue

            # 文本去 HTML
            text = re.sub(r"<[^>]+>", "", mblog.get("text", ""))

            # 图片
            pics: List[str] = []
            pic_infos = mblog.get("pic_infos") or {}
            for _, info in (pic_infos or {}).items():
                u = info.get("url")
                if u:
                    pics.append(u)
            if not pics and mblog.get("pic_ids"):
                for pid in mblog["pic_ids"]:
                    pics.append(f"https://wx1.sinaimg.cn/large/{pid}.jpg")

            # 时间 / 用户
            ts = parse_api_time(mblog.get("created_at", ""))
            user_name = (mblog.get("user") or {}).get("screen_name", "")

            items.append({
                "id": str(wid),
                "text": text,
                "pics": ",".join(pics) if pics else "",
                "timestamp": ts,
                "source": "新浪微博移动端",
                "user_name": user_name,
                "reposts_count": int(mblog.get("reposts_count", 0)),
                "comments_count": int(mblog.get("comments_count", 0)),
                "attitudes_count": int(mblog.get("attitudes_count", 0)),
            })
        return items

    def _comments_once(self, weibo_id: str, max_id: Optional[int] = None) -> Dict[str, Any]:
        """请求一次评论 API（热评），返回 JSON"""
        url = f"{self.BASE}/comments/hotflow"
        params = {
            "id": weibo_id,
            "mid": weibo_id,
            "max_id_type": 0,
        }
        if max_id:
            params["max_id"] = max_id

        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.status_code != 200:
            return {"ok": 0, "msg": f"HTTP {r.status_code}"}
        try:
            return r.json()
        except Exception:
            return {"ok": 0, "msg": "invalid json"}

    def _crawl_comments(self, weibo_id: str) -> List[Dict[str, Any]]:
        if not self.crawl_comments:
            return []
        max_id = None
        page = 0
        results: List[Dict[str, Any]] = []

        while page < self.max_comment_pages:
            data = self._comments_once(weibo_id, max_id=max_id)
            if data.get("ok") != 1:
                break
            rows = (data.get("data") or {}).get("data") or []
            for it in rows:
                cid = str(it.get("id", ""))
                if not cid:
                    continue
                content = it.get("text") or it.get("text_raw") or ""
                content = re.sub(r"<[^>]+>", "", content)
                ts = parse_api_time(it.get("created_at", "")) if it.get("created_at") else int(time.time())
                results.append({
                    "id": cid,
                    "weibo_id": weibo_id,
                    "user": (it.get("user") or {}).get("screen_name", ""),
                    "content": content.strip(),
                    "timestamp": ts,
                })
            page += 1
            max_id = (data.get("data") or {}).get("max_id")
            if not max_id:
                break
            time.sleep(get_delay(self.delay_cfg) / 2.0)
        return results

    # ---------- Main flow ----------
    def crawl_keywords(self) -> bool:
        try:
            keywords: List[str] = list(self.cfg.get("keywords") or [])
            start_page = int(self.cfg.get("startPage", 1))
            max_page = int(self.cfg.get("maxPage", 1))
        except Exception:
            logger.error("配置字段缺失或格式错误（keywords/startPage/maxPage）")
            return False

        logger.info("评论抓取：%s，最多页数：%s", self.crawl_comments, self.max_comment_pages)
        total_weibo = 0
        total_comments = 0

        for kw in keywords:
            logger.info("开始爬取关键词：%s", kw)
            kid = self.db.ensure_keyword_id(kw)
            if not kid:
                logger.error("无法获取/创建关键字ID：%s，跳过该关键词", kw)
                continue

            for page in range(start_page, max_page + 1):
                logger.info("关键词 %s —— 第 %d 页", kw, page)
                items = self._search_page(kw, page)
                if not items:
                    logger.info("关键词 %s —— 第 %d 页无数据，停止翻页", kw, page)
                    break

                saved = 0
                for it in items:
                    if self.db.save_weibo(it, kid):
                        saved += 1
                        # 评论
                        if self.crawl_comments:
                            cmts = self._crawl_comments(it["id"])
                            total_comments += self.db.save_comments(cmts)

                total_weibo += saved
                logger.info("关键词 %s —— 第 %d 页保存 %d 条微博", kw, page, saved)

                delay = get_delay(self.delay_cfg)
                logger.info("等待 %.2f 秒后继续", delay)
                time.sleep(delay)

            logger.info("关键词 %s 爬取结束", kw)

        logger.info("全部关键词完成：微博 %d 条，评论 %d 条", total_weibo, total_comments)
        return True

# ---------------- Entrypoint ----------------
def main():
    print("🚀 微博移动端爬虫启动中...")
    print("=" * 50)
    # 可选：确保表结构
    try:
        ensure_db(load_config().get("mysql_config", {}))
    except Exception:
        pass

    # 简单的数据库连通性提示
    try:
        db = DB(load_config()["mysql_config"])
        db.close()
        print("✅ 数据库连接正常")
    except Exception as e:
        print("❌ 数据库连接失败：", e)
        sys.exit(1)

    try:
        crawler = WeiboMobileCrawler()
        ok = crawler.crawl_keywords()
        print("✅ 爬取完成！" if ok else "❌ 爬取失败")
        sys.exit(0 if ok else 2)
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
        sys.exit(130)
    except Exception as e:
        logger.exception("运行异常：%s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
