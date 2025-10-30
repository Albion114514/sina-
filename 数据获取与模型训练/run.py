#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动脚本（改进版）
- 依赖检查以 PyTorch 为主（不再强制 TensorFlow）
- Redis 变为可选：缺失仅警告，不阻塞
- 清晰输出与退出码
"""

import sys
import subprocess
from pathlib import Path

def check_pkg(name):
    try:
        __import__(name)
        return True, None
    except Exception as e:
        return False, str(e)

def print_ok(msg): print(f"✅ {msg}")
def print_warn(msg): print(f"⚠️  {msg}")
def print_err(msg): print(f"❌ {msg}")
def print_bar(title): 
    print("\n" + "="*60)
    print(title)
    print("="*60)

def check_dependencies():
    print_bar("依赖检查")
    hard_ok = True

    # Hard deps
    for mod, alias in [("flask","Flask"), ("flask_sqlalchemy","Flask-SQLAlchemy"), ("pymysql","PyMySQL"), ("torch","PyTorch")]:
        ok, err = check_pkg(mod)
        if ok:
            print_ok(f"{alias} 已安装")
        else:
            print_err(f"{alias} 未安装: {err}")
            hard_ok = False

    # Optional deps
    for mod, alias in [("redis","Redis")]:
        ok, err = check_pkg(mod)
        if ok:
            print_ok(f"{alias} 可用（可选）")
        else:
            print_warn(f"{alias} 未安装（可选）: {err}")

    return hard_ok

def find_app_entry():
    for p in ["app.py", "run_app.py", "main.py"]:
        if Path(p).exists():
            return p
    return None

def start_backend():
    entry = find_app_entry()
    if not entry:
        print_err("未找到后端入口（app.py / run_app.py / main.py）")
        sys.exit(2)
    print_bar(f"启动后端: {entry}")
    # 直接转发到 Python 子进程
    p = subprocess.Popen([sys.executable, entry])
    try:
        p.wait()
        code = p.returncode or 0
        if code == 0:
            print_ok("后端正常退出")
        else:
            print_err(f"后端异常退出，返回码={code}")
        sys.exit(code)
    except KeyboardInterrupt:
        p.terminate()
        print_warn("收到中断信号，正在退出...")
        sys.exit(0)

def main():
    good = check_dependencies()
    if not good:
        print_err("必需依赖缺失，请先安装后重试。")
        sys.exit(1)
    start_backend()

if __name__ == "__main__":
    main()
