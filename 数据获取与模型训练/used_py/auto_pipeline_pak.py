#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_pipeline_fixed.py
修复版本 - 改进用户界面，更清晰的开局选项
"""

import os
import sys
import json
import time
import csv
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Optional, List, Tuple

# ---------- Constants ----------
TRAIN_OUTPUT_DIR = "train_output"
TEST_OUTPUT_DIR = "test_output"

WEIBO_TO_CSV_SCRIPT_CANDIDATES = ["weibo_to_csv.py"]
CRAWLER_SCRIPT_CANDIDATES = [
    "weibo_mobile_crawler_main.py",
    "weibo_crawler_2_1_main.py",
    "weibo_crawler_main.py",
]
ALGO_SCRIPT_CANDIDATES = ["rumor_detect.py"]

CONFIG_PATH = "config.json"


# ---------- Utilities ----------

def ensure_module_or_none(modname: str):
    """Try to import a module; if missing, return None."""
    try:
        return __import__(modname)
    except Exception:
        return None


settings = ensure_module_or_none("settings")


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load config.json; if not present, raise clear error."""
    if settings and hasattr(settings, "load_config"):
        return settings.load_config(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sqlalchemy_uri(cfg: dict) -> str:
    """Return a SQLAlchemy URI built from cfg['mysql_config'] (fallback if settings absent)."""
    if settings and hasattr(settings, "build_sqlalchemy_uri"):
        return settings.build_sqlalchemy_uri(cfg)
    mc = cfg["mysql_config"].copy()
    user = mc.get("user", "root")
    pwd = mc.get("password", "")
    host = mc.get("host", "localhost")
    port = int(mc.get("port", 3306))
    db = mc.get("database", "")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"


def run_subprocess(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            cwd=cwd
        )
        return p.returncode, p.stdout, p.stderr
    except UnicodeDecodeError:
        # 如果utf-8失败，尝试其他编码
        p = subprocess.run(
            cmd,
            capture_output=True,
            cwd=cwd
        )
        stdout = p.stdout.decode('utf-8', errors='ignore')
        stderr = p.stderr.decode('utf-8', errors='ignore')
        return p.returncode, stdout, stderr


def pick_existing(paths: List[str]) -> Optional[str]:
    """Return the first existing path from candidates."""
    for p in paths:
        if Path(p).exists():
            return p
    return None


def slugify(text: str, max_len: int = 32) -> str:
    """Safe slug from text for filesystem names."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in text.strip())
    if not safe:
        safe = "topic"
    return safe[:max_len].strip("_")


def print_header(title: str):
    """打印美观的标题"""
    print("\n" + "=" * 60)
    print(f"📋 {title}")
    print("=" * 60)


def print_step(step_num: int, description: str):
    """打印步骤信息"""
    print(f"\n🔹 步骤 {step_num}: {description}")


def print_success(message: str):
    """打印成功信息"""
    print(f"✅ {message}")


def print_warning(message: str):
    """打印警告信息"""
    print(f"⚠️ {message}")


def print_error(message: str):
    """打印错误信息"""
    print(f"❌ {message}")


def print_info(message: str):
    """打印一般信息"""
    print(f"ℹ️ {message}")


# ---------- DB Helpers (pymysql) ----------

def get_db_conn(cfg: dict):
    import pymysql
    mc = cfg["mysql_config"].copy()
    mc.setdefault("charset", "utf8mb4")
    mc["port"] = int(mc.get("port", 3306))
    return pymysql.connect(**mc)


def bootstrap_schema_if_possible():
    """Run db_bootstrap.py --mode init if it exists; else no-op."""
    if Path("db_bootstrap.py").exists():
        print_step(1, "初始化数据库表结构")
        rc, out, err = run_subprocess([sys.executable, "db_bootstrap.py", "--mode", "init"])
        if rc != 0:
            print_warning("数据库初始化失败 (继续执行):")
            print(err or out)


def recreate_database_tables(cfg: dict) -> bool:
    """重新创建所有数据库表"""
    try:
        # 尝试使用现有的修复脚本
        if Path("db_setup_fixed.py").exists():
            print_info("使用 db_setup_fixed.py 重建表...")
            rc, out, err = run_subprocess([sys.executable, "db_setup_fixed.py"])
            if rc == 0:
                print_success("表重建成功")
                return True
            else:
                print_error("表重建失败")
                return False
        elif Path("db_setup2_0.py").exists():
            print_info("使用 db_setup2_0.py 重建表...")
            rc, out, err = run_subprocess([sys.executable, "db_setup2_0.py"])
            if rc == 0:
                print_success("表重建成功")
                return True
            else:
                print_error("表重建失败")
                return False
        else:
            print_warning("未找到数据库初始化脚本，使用内置方法重建...")
            return recreate_tables_manual(cfg)
    except Exception as e:
        print_error(f"表重建失败: {e}")
        return False


def recreate_tables_manual(cfg: dict) -> bool:
    """手动重建表结构"""
    try:
        conn = get_db_conn(cfg)
        cursor = conn.cursor()

        print_info("删除外键约束...")
        cursor.execute("""
            SELECT CONSTRAINT_NAME, TABLE_NAME 
            FROM information_schema.KEY_COLUMN_USAGE 
            WHERE TABLE_SCHEMA = %s 
            AND REFERENCED_TABLE_NAME IS NOT NULL
        """, (cfg['mysql_config']['database'],))

        foreign_keys = cursor.fetchall()
        for fk_name, table_name in foreign_keys:
            try:
                cursor.execute(f"ALTER TABLE {table_name} DROP FOREIGN KEY {fk_name}")
                print_success(f"删除外键: {table_name}.{fk_name}")
            except Exception as e:
                print_warning(f"删除外键失败 {table_name}.{fk_name}: {e}")

        # 删除表（按依赖顺序）
        tables_to_drop = [
            'topic_keywords', 'prediction_results', 'comments',
            'weibo_data', 'crawl_log', 'keywords', 'topics'
        ]

        for table in tables_to_drop:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                print_success(f"删除表: {table}")
            except Exception as e:
                print_warning(f"删除表失败 {table}: {e}")

        print_info("创建新表...")
        # 创建关键字表
        cursor.execute("""
            CREATE TABLE keywords (
                id INT AUTO_INCREMENT PRIMARY KEY,
                keyword VARCHAR(255) NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_keyword (keyword)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print_success("创建关键字表")

        # 创建微博数据表
        cursor.execute("""
            CREATE TABLE weibo_data (
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
                FOREIGN KEY (keyword_id) REFERENCES keywords(id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print_success("创建微博数据表")

        # 在 recreate_tables_manual() 里、创建 weibo_data 之后，继续加上：

        # comments
        cursor.execute("""
            CREATE TABLE comments (
                id VARCHAR(50) PRIMARY KEY,
                weibo_id VARCHAR(50) NOT NULL,
                user VARCHAR(100),
                content TEXT NOT NULL,
                timestamp BIGINT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_weibo_id (weibo_id),
                FOREIGN KEY (weibo_id) REFERENCES weibo_data(id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # crawl_log
        cursor.execute("""
            CREATE TABLE crawl_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                keyword VARCHAR(255) NOT NULL,
                page_num INT NOT NULL,
                crawl_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success_count INT DEFAULT 0,
                error_message TEXT,
                INDEX idx_keyword (keyword)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # topics
        cursor.execute("""
            CREATE TABLE topics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                mode ENUM('train','test') NOT NULL,
                rumor_type TINYINT,
                UNIQUE KEY uk_title (title)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # topic_keywords
        cursor.execute("""
            CREATE TABLE topic_keywords (
                id INT AUTO_INCREMENT PRIMARY KEY,
                topic_id INT NOT NULL,
                keyword_id INT NOT NULL,
                UNIQUE KEY uk_topic_keyword (topic_id, keyword_id),
                FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE,
                FOREIGN KEY (keyword_id) REFERENCES keywords(id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # prediction_results
        cursor.execute("""
            CREATE TABLE prediction_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                topic_id INT NOT NULL,
                prediction TEXT NOT NULL,
                accuracy FLOAT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_topic (topic_id),
                FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # 插入默认关键字
        keywords = cfg.get('keywords', [])
        for keyword in keywords:
            cursor.execute("INSERT IGNORE INTO keywords (keyword) VALUES (%s)", (keyword,))
        print_success(f"插入 {len(keywords)} 个关键字")

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print_error(f"手动重建表失败: {e}")
        return False


# ---------- Core Steps ----------

def get_user_choices(cfg: dict):
    """获取用户的所有选择"""
    print_header("微博谣言检测自动化系统")
    print("欢迎使用自动化流程！请按照以下步骤进行配置：")

    # 步骤1: 数据库表重建选择
    print_step(1, "数据库表配置")
    print("当前数据库状态：")
    print("  - 如果首次运行或需要重置数据，请选择重建")
    print("  - 如果已有数据并希望保留，请选择使用现有表")

    recreate_choice = None
    while recreate_choice not in ("y", "n", "Y", "N"):
        recreate_choice = input("是否重新创建数据库表？(y/n): ").strip().lower()

    # 步骤2: 关键字确认
    print_step(2, "关键字配置")
    current_keywords = cfg.get('keywords', [])
    print(f"当前配置的关键字: {current_keywords}")
    print("这些关键字将用于微博搜索")

    keyword_choice = None
    while keyword_choice not in ("y", "n", "Y", "N"):
        keyword_choice = input("确认使用这些关键字？(y/n): ").strip().lower()

    if keyword_choice != "y":
        print("请修改 config.json 文件中的 keywords 配置后重新运行程序。")
        sys.exit(1)

    # 步骤3: 模式选择
    print_step(3, "运行模式选择")
    print("请选择运行模式：")
    print("  - 训练模式 (1): 用于训练模型，需要指定内容类型")
    print("  - 测试模式 (2): 用于测试模型，不需要指定内容类型")

    mode = None
    while mode not in ("1", "2"):
        mode = input("请选择模式 (1:训练, 2:测试): ").strip()
    mode = "train" if mode == "1" else "test"

    # 步骤4: 如果是训练模式，选择内容类型
    rumor_type = None
    if mode == "train":
        print_step(4, "训练内容类型选择")
        print("请选择训练数据的类型：")
        print("  - 谣言 (1): 标记为谣言的数据")
        print("  - 非谣言 (2): 标记为非谣言的数据")

        rt = None
        while rt not in ("1", "2"):
            rt = input("请选择内容类型 (1:谣言, 2:非谣言): ").strip()
        rumor_type = int(rt)

    # 步骤5: 输入大标题
    print_step(5, "主题命名")
    print("请输入本次任务的大标题：")
    print("  - 这将用于创建输出目录和数据库记录")
    print("  - 建议使用有意义的名称，如'红军城大捷谣言检测'")

    topic_title = ""
    while not topic_title.strip():
        topic_title = input("请输入大标题: ").strip()
        if not topic_title.strip():
            print_warning("大标题不能为空，请重新输入")

    return {
        'recreate_tables': recreate_choice == 'y',
        'keywords_confirmed': keyword_choice == 'y',
        'mode': mode,
        'rumor_type': rumor_type,
        'topic_title': topic_title,
        'keywords': current_keywords
    }


def ensure_keywords(cur, keywords: List[str]) -> List[int]:
    """Ensure keywords exist; return their IDs in order."""
    ids: List[int] = []
    for k in keywords:
        cur.execute("SELECT id FROM keywords WHERE keyword=%s", (k,))
        row = cur.fetchone()
        if row:
            ids.append(int(row[0]))
        else:
            cur.execute("INSERT INTO keywords (keyword) VALUES (%s)", (k,))
            ids.append(int(cur.lastrowid))
    return ids


def save_topic_and_link_keywords(conn, title: str, mode: str, rumor_type: Optional[int], kw_list: List[str]) -> int:
    """Insert or update topic, link to keywords in topic_keywords; return topic_id."""
    with conn.cursor() as cur:
        # upsert topic
        cur.execute(
            """
            INSERT INTO topics (title, mode, rumor_type)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE mode=VALUES(mode), rumor_type=VALUES(rumor_type)
            """,
            (title, mode, rumor_type),
        )
        if cur.lastrowid:
            topic_id = int(cur.lastrowid)
        else:
            cur.execute("SELECT id FROM topics WHERE title=%s", (title,))
            topic_id = int(cur.fetchone()[0])

        # ensure keywords and link
        kw_ids = ensure_keywords(cur, kw_list)
        for kid in kw_ids:
            cur.execute(
                """
                INSERT IGNORE INTO topic_keywords (topic_id, keyword_id)
                VALUES (%s, %s)
                """,
                (topic_id, kid),
            )
    conn.commit()
    return topic_id


def run_crawler():
    """Run the crawler script (pick the first that exists)."""
    crawler = pick_existing(CRAWLER_SCRIPT_CANDIDATES)
    if not crawler:
        print_warning("未找到爬虫脚本，请将爬虫脚本命名为以下之一并放在当前目录：")
        for c in CRAWLER_SCRIPT_CANDIDATES:
            print("  -", c)
        return False

    print_step(6, "运行微博爬虫")
    print_info(f"使用爬虫脚本: {crawler}")
    rc, out, err = run_subprocess([sys.executable, crawler])

    if rc != 0:
        print_error("爬虫运行失败：")
        print(err or out)
        return False

    print_success("爬虫运行完成")
    print(out)
    return True


def create_unique_output_dir(mode: str, topic_title: str) -> Path:
    base = TRAIN_OUTPUT_DIR if mode == "train" else TEST_OUTPUT_DIR
    Path(base).mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(topic_title, max_len=24)
    unique = f"{stamp}_{slug}_{uuid4().hex[:6]}"
    outdir = Path(base) / unique
    outdir.mkdir(parents=True, exist_ok=False)
    print_success(f"输出目录: {outdir}")
    return outdir


def run_weibo_to_csv(outdir: Path) -> bool:
    """Run weibo_to_csv export into outdir; expects script exists."""
    tool = pick_existing(WEIBO_TO_CSV_SCRIPT_CANDIDATES)
    if not tool:
        print_warning("未找到 weibo_to_csv.py，请将脚本放在当前目录")
        return False

    print_step(7, "导出数据到CSV")
    print_info(f"使用导出脚本: {tool}")
    rc, out, err = run_subprocess([sys.executable, tool, "--output", str(outdir)])

    if rc != 0:
        print_error("导出失败：")
        print(err or out)
        return False

    print_success("数据导出完成")
    print(out)
    return True


def run_algorithm(mode: str, rumor_type: Optional[int], outdir: Path) -> Tuple[bool, Optional[float]]:
    """
    Run algorithm component on generated CSVs.
    Expected inputs:
      - weibo_data_format.csv
      - comments_format.csv
    Output:
      - prediction_result.csv
    Return (ok, accuracy) where accuracy can be None if not found.
    """
    algo = pick_existing(ALGO_SCRIPT_CANDIDATES)
    if not algo:
        print_warning("未找到算法脚本 rumor_detect.py；将跳过算法步骤。")
        return False, None

    content = outdir / "weibo_data_format.csv"
    comments = outdir / "comments_format.csv"
    output = outdir / "prediction_result.csv"

    if not content.exists() or not comments.exists():
        print_warning("算法输入 CSV 缺失：需要 weibo_data_format.csv 和 comments_format.csv")
        return False, None

    cmd = [
        sys.executable, str(algo),
        "--content", str(content),
        "--comments", str(comments),
        "--output", str(output),
        "--mode", "train" if mode == "train" else "test"
    ]
    if mode == "train" and rumor_type in (1, 2):
        cmd.extend(["--label", str(rumor_type)])

    print_step(8, "运行谣言检测算法")
    print_info("运行命令: " + " ".join(cmd))
    rc, out, err = run_subprocess(cmd)

    if rc != 0:
        print_error("算法运行失败：")
        print(err or out)
        return False, None

    # Try parse accuracy from stdout (e.g., "Accuracy: 0.89")
    acc = None
    for line in (out or "").splitlines():
        if "Accuracy" in line:
            try:
                acc = float(line.split(":")[-1].strip())
            except Exception:
                pass

    print_success("算法运行完成")
    print(out)
    return True, acc


def save_prediction(conn, topic_id: int, outdir: Path, accuracy: Optional[float]) -> bool:
    """Read prediction_result.csv content and store into prediction_results table (1:1 topic)."""
    pred_path = outdir / "prediction_result.csv"
    if not pred_path.exists():
        print_warning("找不到 prediction_result.csv，写库将跳过")
        return False

    content = pred_path.read_text(encoding="utf-8", errors="ignore")

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO prediction_results (topic_id, prediction, accuracy)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                prediction=VALUES(prediction),
                accuracy=VALUES(accuracy)
            """,
            (topic_id, content, accuracy),
        )
    return True


def write_topic_info_csv(conn, topic_id: int, outdir: Path) -> None:
    """Write topic_info.csv with topic + prediction join into outdir."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.title, t.mode, t.rumor_type, p.prediction, p.accuracy, t.created_at
            FROM topics t
            LEFT JOIN prediction_results p ON t.id = p.topic_id
            WHERE t.id = %s
            """,
            (topic_id,),
        )
        row = cur.fetchone()

    csv_path = outdir / "topic_info.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "mode", "rumor_type", "prediction", "accuracy", "created_at"])
        if row:
            writer.writerow(row)


def main():
    # 加载配置
    cfg = load_config()

    # 获取用户选择
    choices = get_user_choices(cfg)

    # 数据库表处理
    if choices['recreate_tables']:
        print_info("重新创建数据库表...")
        if recreate_database_tables(cfg):
            print_success("数据库表重建完成")
        else:
            print_error("数据库表重建失败，继续使用现有表")
    else:
        print_info("使用现有数据库表")
        bootstrap_schema_if_possible()

    # 数据库连接
    conn = get_db_conn(cfg)
    try:
        # 保存主题和关联关键字
        print_step(9, "保存主题信息到数据库")
        topic_id = save_topic_and_link_keywords(
            conn,
            choices['topic_title'],
            choices['mode'],
            choices['rumor_type'],
            choices['keywords']
        )
        print_success(f"主题保存成功，ID: {topic_id}")

        # 运行爬虫
        if not run_crawler():
            yn = input("爬虫运行失败，是否继续后续步骤? (y/n): ").strip().lower()
            if yn != "y":
                sys.exit(1)

        # 创建输出目录
        outdir = create_unique_output_dir(choices['mode'], choices['topic_title'])

        # 导出CSV
        if not run_weibo_to_csv(outdir):
            yn = input("导出失败，是否继续算法步骤? (y/n): ").strip().lower()
            if yn != "y":
                sys.exit(1)

        # 运行算法
        ok_algo, acc = run_algorithm(choices['mode'], choices['rumor_type'], outdir)
        if not ok_algo:
            print_warning("算法未成功执行，数据库记录将不包含预测内容/准确率")

        # 保存预测结果到数据库
        print_step(10, "保存预测结果到数据库")
        if save_prediction(conn, topic_id, outdir, acc):
            print_success("预测结果保存成功")
        else:
            print_warning("预测结果保存失败")

        # 写入topic_info.csv
        print_step(11, "生成主题信息文件")
        write_topic_info_csv(conn, topic_id, outdir)
        print_success("主题信息文件生成完成")

        # 显示完成信息
        print_header("流程完成")
        print_success(f"所有操作已完成！")
        print(f"📁 输出目录: {outdir}")
        print(f"🔍 主题标题: {choices['topic_title']}")
        print(f"📊 运行模式: {choices['mode']}")
        if choices['mode'] == 'train':
            print(f"🏷️  内容类型: {'谣言' if choices['rumor_type'] == 1 else '非谣言'}")
        print(f"🔑 使用关键字: {', '.join(choices['keywords'])}")
        print("\n🎉 您现在可以查看输出目录中的结果文件！")

    except Exception as e:
        print_error(f"流程执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


if __name__ == "__main__":
    main()