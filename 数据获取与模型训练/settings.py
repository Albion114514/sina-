# settings.py
import json
from pathlib import Path

def load_config(path: str = "config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_sqlalchemy_uri(cfg: dict) -> str:
    mc = cfg["mysql_config"].copy()
    user = mc.get("user", "root")
    pwd = mc.get("password", "")
    host = mc.get("host", "localhost")
    port = int(mc.get("port", 3307))
    db   = mc.get("database", "")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
