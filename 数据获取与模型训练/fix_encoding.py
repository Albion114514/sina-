#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_encoding.py - 修复项目编码问题
"""

import os
import sys


def set_utf8_environment():
    """设置UTF-8环境变量"""
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'

    # 尝试重新配置标准输出
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # 旧版本Python不支持reconfigure

    print("✅ UTF-8环境已设置")


if __name__ == "__main__":
    set_utf8_environment()