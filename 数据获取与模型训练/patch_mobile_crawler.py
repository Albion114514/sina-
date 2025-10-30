#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_mobile_crawler.py
Applies safe, in-place edits to `weibo_mobile_crawler_main.py`:

1) Remove invalid imports/calls to db_setup_fixed / db_setup2_0 / db_bootstrap.* functions
   that don't exist in your project layout (they can cause early exits).
2) Optionally inject lightweight debug prints around the mobile API response
   (enabled when env DEBUG_WEIBO=1).

Usage:
    python patch_mobile_crawler.py
This will update the file in-place and create a `.bak` backup next to it.
"""

import re
from pathlib import Path

TARGET = Path("weibo_mobile_crawler_main.py")
BACKUP_SUFFIX = ".bak"

def patch_file():
    if not TARGET.exists():
        print("❌ 未找到", TARGET)
        return 1

    src = TARGET.read_text(encoding="utf-8", errors="ignore")

    original = src

    # --- 1) Comment out invalid imports and calls ---
    # - from db_setup_fixed import fix_database
    src = re.sub(
        r'^\s*from\s+db_setup_fixed\s+import\s+fix_database\s*\n', 
        '# (patched) from db_setup_fixed import fix_database\n', 
        src, flags=re.MULTILINE
    )
    # - if fix_database(): ...  (simple conservative patch: comment the whole line)
    src = re.sub(
        r'^\s*if\s+fix_database\(\)\s*:\s*\n',
        '# (patched) if fix_database():\n',
        src, flags=re.MULTILINE
    )
    # Also comment the immediate print lines that follow the above (best-effort)
    src = src.replace("print(\"✅ 数据库修复完成\")", "# (patched) print(\"✅ 数据库修复完成\")")
    src = src.replace("print(\"⚠️ 数据库修复失败，但继续执行\")", "# (patched) print(\"⚠️ 数据库修复失败，但继续执行\")")

    # - from db_setup2_0 import create_tables, load_config
    src = re.sub(
        r'^\s*from\s+db_setup2_0\s+import\s+create_tables,\s*load_config\s*\n',
        '# (patched) from db_setup2_0 import create_tables, load_config\n',
        src, flags=re.MULTILINE
    )
    # Any create_tables(…) / load_config(…) calls in main() — comment conservatively
    src = re.sub(
        r'^\s*create_tables\([^)]*\)\s*$', 
        '# (patched) create_tables(...)', 
        src, flags=re.MULTILINE
    )
    src = re.sub(
        r'^\s*config\s*=\s*load_config\([^)]*\)\s*$', 
        '# (patched) config = load_config(...)', 
        src, flags=re.MULTILINE
    )

    # --- 2) Inject optional debug prints (guarded by env var) ---
    # We'll insert a small helper near the top if not present.
    if "def _dbg(" not in src:
        helper = (
            "\n\n# (patched) debug helper\n"
            "import os as _os\n"
            "def _dbg(*a, **k):\n"
            "    if _os.environ.get('DEBUG_WEIBO') == '1':\n"
            "        try:\n"
            "            print('[DEBUG]', *a, **k)\n"
            "        except Exception:\n"
            "            pass\n"
        )
        # place helper after imports block (best-effort: find first blank line after imports)
        m = re.search(r'(\n\s*\n)', src)
        if m:
            idx = m.end()
            src = src[:idx] + helper + src[idx:]
        else:
            src = helper + src

    # Add debug near API call response in crawl_weibo_search_api (best-effort locate)
    src = src.replace(
        "data = response.json()",
        "data = response.json()\n        _dbg('status', response.status_code, 'ok', data.get('ok'), 'msg', data.get('msg'))"
    )

    if src == original:
        print("ℹ 未检测到需要修改的内容，文件保持不变。")
        return 0

    # Backup and write
    backup = TARGET.with_suffix(TARGET.suffix + BACKUP_SUFFIX)
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(src, encoding="utf-8")
    print(f"✅ 已修补 {TARGET}（备份: {backup.name}）")
    return 0

if __name__ == "__main__":
    raise SystemExit(patch_file())