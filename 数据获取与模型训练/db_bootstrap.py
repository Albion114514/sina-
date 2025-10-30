# db_bootstrap.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import pymysql
from settings import load_config

TABLES_CORE = ["keywords", "weibo_data", "comments", "crawl_log"]
TABLES_EXTRA = ["topics", "topic_keywords", "prediction_results"]

SQLS_CREATE = {
    "keywords": """
        CREATE TABLE IF NOT EXISTS keywords (
            id INT AUTO_INCREMENT PRIMARY KEY,
            keyword VARCHAR(255) NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_keyword (keyword)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "weibo_data": """
        CREATE TABLE IF NOT EXISTS weibo_data (
            id VARCHAR(50) PRIMARY KEY,
            keyword_id INT NOT NULL,
            text TEXT NOT NULL,
            pics TEXT,
            timestamp BIGINT NOT NULL,
            source VARCHAR(100) DEFAULT '新浪微博移动端',
            user_name VARCHAR(255) DEFAULT '',
            reposts_count INT DEFAULT 0,
            comments_count INT DEFAULT 0,
            attitudes_count INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_timestamp (timestamp),
            INDEX idx_keyword_id (keyword_id),
            CONSTRAINT fk_weibo_keyword
              FOREIGN KEY (keyword_id) REFERENCES keywords(id)
              ON DELETE RESTRICT ON UPDATE RESTRICT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "comments": """
        CREATE TABLE IF NOT EXISTS comments (
            id VARCHAR(50) PRIMARY KEY,
            weibo_id VARCHAR(50) NOT NULL,
            user VARCHAR(100),
            content TEXT NOT NULL,
            timestamp BIGINT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_weibo_id (weibo_id),
            CONSTRAINT fk_comment_weibo
              FOREIGN KEY (weibo_id) REFERENCES weibo_data(id)
              ON DELETE CASCADE ON UPDATE RESTRICT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "crawl_log": """
        CREATE TABLE IF NOT EXISTS crawl_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            keyword VARCHAR(255) NOT NULL,
            page_num INT NOT NULL,
            crawl_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success_count INT DEFAULT 0,
            error_message TEXT,
            INDEX idx_keyword (keyword)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "topics": """
        CREATE TABLE IF NOT EXISTS topics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            mode ENUM('train','test') NOT NULL,
            rumor_type TINYINT,
            UNIQUE KEY uk_title (title)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "topic_keywords": """
        CREATE TABLE IF NOT EXISTS topic_keywords (
            id INT AUTO_INCREMENT PRIMARY KEY,
            topic_id INT NOT NULL,
            keyword_id INT NOT NULL,
            UNIQUE KEY uk_topic_keyword (topic_id, keyword_id),
            CONSTRAINT fk_tk_topic FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE,
            CONSTRAINT fk_tk_keyword FOREIGN KEY (keyword_id) REFERENCES keywords(id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "prediction_results": """
        CREATE TABLE IF NOT EXISTS prediction_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            topic_id INT NOT NULL,
            prediction TEXT NOT NULL,
            accuracy FLOAT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uk_topic (topic_id),
            CONSTRAINT fk_pr_topic FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
}


def drop_all(cursor, dbname: str):
    cursor.execute("""
        SELECT CONSTRAINT_NAME, TABLE_NAME 
        FROM information_schema.KEY_COLUMN_USAGE 
        WHERE TABLE_SCHEMA=%s AND REFERENCED_TABLE_NAME IS NOT NULL
    """, (dbname,))
    for fk, tbl in cursor.fetchall():
        try:
            cursor.execute(f"ALTER TABLE `{tbl}` DROP FOREIGN KEY `{fk}`")
        except:
            pass
    for tbl in (["topic_keywords", "prediction_results", "comments", "weibo_data", "crawl_log", "keywords", "topics"]):
        cursor.execute(f"DROP TABLE IF EXISTS `{tbl}`")


def ensure_keywords_seed(cursor, keywords):
    for kw in keywords:
        cursor.execute("INSERT IGNORE INTO keywords (keyword) VALUES (%s)", (kw,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["init", "rebuild"], default="init")
    args = ap.parse_args()

    cfg = load_config()
    mc = cfg["mysql_config"].copy()
    mc.setdefault("charset", "utf8mb4")
    conn = pymysql.connect(**mc)
    cur = conn.cursor()

    if args.mode == "rebuild":
        drop_all(cur, mc["database"])
    # core
    for t in TABLES_CORE:
        cur.execute(SQLS_CREATE[t])
    # extra
    for t in TABLES_EXTRA:
        cur.execute(SQLS_CREATE[t])

    ensure_keywords_seed(cur, cfg.get("keywords", []))
    conn.commit()
    conn.close()
    print(f"✅ db_bootstrap done with mode={args.mode}")


if __name__ == "__main__":
    main()
