"""
logger.py - 结构化日志模块

统一日志格式，所有模块必须通过 get_logger() 获取 logger，禁止自行配置。
"""

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)-15s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _init_root() -> None:
    """初始化根 logger，只执行一次。"""
    global _initialized
    if _initialized:
        return

    root = logging.getLogger("kb_metadata")
    root.setLevel(logging.INFO)
    root.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    log_dir = Path(__file__).resolve().parent.parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取子 logger。name 不含 'kb_metadata.' 前缀。"""
    _init_root()
    return logging.getLogger(f"kb_metadata.{name}")