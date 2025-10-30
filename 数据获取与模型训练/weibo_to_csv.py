#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weibo_to_csv.py
导出微博与评论为 CSV：
1) 原始导出：weibo_data.csv / comments.csv（带表头，便于人工查看/核对）
2) 格式化导出（团队旧规范，供算法读取）：
   - weibo_data_format.csv（无表头，固定 5 列：content_id, keyword, content_text, reserved, timestamp）
   - comments_format.csv（无表头，固定 5 列：comment_id, content_id, comment_name, comment_text, timestamp）

使用：
    python weibo_to_csv.py --output <输出目录>
"""

import os
import csv
import json
import argparse

import pymysql

CONFIG_PATH = "config.json"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def connect_db(mysql_config):
    cfg = mysql_config.copy()
    cfg.setdefault("charset", "utf8mb4")
    cfg["port"] = int(cfg.get("port", 3306))
    return pymysql.connect(**cfg)


def export_raw(conn, outdir):
    """
    原始导出（带表头，便于人工核对，不参与算法）
    - weibo_data.csv
    - comments.csv
    """
    weibo_count = 0
    comments_count = 0

    with conn.cursor() as cur:
        # 微博原始数据
        cur.execute("""
            SELECT
                w.id,
                k.keyword,
                w.text,
                w.pics,
                w.timestamp,
                w.source,
                w.user_name,
                w.reposts_count,
                w.comments_count,
                w.attitudes_count,
                w.created_at,
                w.updated_at
            FROM weibo_data w
            JOIN keywords k ON w.keyword_id = k.id
            ORDER BY w.timestamp DESC
        """)
        rows = cur.fetchall()
        weibo_count = len(rows)
        path_weibo = os.path.join(outdir, "weibo_data.csv")
        with open(path_weibo, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([
                "id","keyword","text","pics","timestamp","source","user_name",
                "reposts_count","comments_count","attitudes_count","created_at","updated_at"
            ])
            for r in rows:
                # 将 timestamp 明确为 int，避免科学计数法
                r = list(r)
                try:
                    r[4] = int(r[4] or 0)
                except Exception:
                    pass
                wr.writerow(r)

        # 评论原始数据（确保 weibo_id 字段存在）
        cur.execute("""
            SELECT
                c.id,
                c.weibo_id,
                c.user,
                c.content,
                c.timestamp,
                c.created_at,
                c.updated_at
            FROM comments c
            ORDER BY c.timestamp DESC
        """)
        rows = cur.fetchall()
        comments_count = len(rows)
        path_comments = os.path.join(outdir, "comments.csv")
        with open(path_comments, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["id","weibo_id","user","content","timestamp","created_at","updated_at"])
            for r in rows:
                r = list(r)
                try:
                    r[4] = int(r[4] or 0)
                except Exception:
                    pass
                wr.writerow(r)

    return weibo_count, comments_count


def export_formatted(conn, outdir):
    """
    格式化导出（旧规范，无表头、固定 5 列），供算法读取：
      weibo_data_format.csv:
        0 content_id
        1 keyword
        2 content_text
        3 reserved            (占位，可写 user_name 或留空)
        4 timestamp (int 秒)
      comments_format.csv:
        0 comment_id
        1 content_id (weibo_id)
        2 comment_name
        3 comment_text
        4 timestamp (int 秒)
    """
    with conn.cursor() as cur:
        # 内容格式化
        cur.execute("""
            SELECT
                w.id            AS content_id,
                k.keyword       AS keyword,
                w.text          AS content_text,
                IFNULL(w.user_name,'') AS reserved,
                w.timestamp     AS ts
            FROM weibo_data w
            JOIN keywords k ON w.keyword_id = k.id
            ORDER BY w.timestamp DESC
        """)
        rows = cur.fetchall()
        path_content = os.path.join(outdir, "weibo_data_format.csv")
        with open(path_content, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            for r in rows:
                cid, kw, text, reserved, ts = r
                try:
                    ts = int(ts or 0)
                except Exception:
                    ts = 0
                wr.writerow([cid, kw, text, reserved, ts])

        # 评论格式化
        cur.execute("""
            SELECT
                c.id        AS comment_id,
                c.weibo_id  AS content_id,
                IFNULL(c.`user`,'') AS comment_name,
                c.content   AS comment_text,
                c.timestamp AS ts
            FROM comments c
            ORDER BY c.timestamp DESC
        """)
        rows = cur.fetchall()
        path_comments = os.path.join(outdir, "comments_format.csv")
        with open(path_comments, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            for r in rows:
                cmid, wid, name, text, ts = r
                try:
                    ts = int(ts or 0)
                except Exception:
                    ts = 0
                wr.writerow([cmid, wid, name, text, ts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="输出目录")
    args = ap.parse_args()

    outdir = args.output
    os.makedirs(outdir, exist_ok=True)

    cfg = load_config()
    conn = connect_db(cfg["mysql_config"])
    try:
        # 1) 原始导出
        wcnt, ccnt = export_raw(conn, outdir)
        print(f"✅ 原始微博数据导出完成：{os.path.join(outdir, 'weibo_data.csv')}（{wcnt}条）")
        print(f"✅ 原始评论数据导出完成：{os.path.join(outdir, 'comments.csv')}（{ccnt}条）")

        # 2) 格式化导出（旧规范，无表头、固定 5 列）
        export_formatted(conn, outdir)
        print(f"✅ 格式化微博数据导出完成：{os.path.join(outdir, 'weibo_data_format.csv')}")
        print(f"✅ 格式化评论数据导出完成：{os.path.join(outdir, 'comments_format.csv')}")
        print("\n🎉 所有数据导出完成！")

    finally:
        conn.close()


if __name__ == "__main__":
    main()