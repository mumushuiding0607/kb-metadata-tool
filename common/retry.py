"""
retry.py - 通用重试装饰器

业务模块只调用 with_retry()，不自行实现重试循环。
"""

import time
from typing import Callable, TypeVar

from common.logger import get_logger
from common.model_interface import TransientError

logger = get_logger("retry")
T = TypeVar("T")


def with_retry(fn: Callable[[], T],
               delays: list[int] | None = None,
               description: str = "") -> T:
    """执行 fn，捕获 TransientError 后按 delays 间隔重试，最终失败抛出。

    Args:
        fn: 待执行函数
        delays: 重试间隔秒数列表（如 [5, 10] 表示最多重试 2 次）
        description: 用于日志的描述
    """
    delays = delays or [5, 10]
    desc = description or "操作"
    last_err: Exception | None = None
    for attempt in range(len(delays) + 1):
        try:
            return fn()
        except TransientError as e:
            last_err = e
            if attempt < len(delays):
                wait = delays[attempt]
                logger.warning(
                    "llm_retry desc=%s attempt=%d/%d wait=%ds err_type=%s err=%s",
                    desc, attempt + 1, len(delays) + 1, wait,
                    type(e).__name__, e,
                )
                time.sleep(wait)
    logger.error("llm_retry_exhausted desc=%s attempts=%d last_err=%s",
                 desc, len(delays) + 1, last_err)
    raise TransientError(f"{desc} 重试 {len(delays) + 1} 次仍失败: {last_err}")