"""
token_bucket.py - 令牌桶限流器

按 rpm_limit 自动补充令牌，避免请求成簇触发 API 限流。
支持突发（capacity 内允许瞬时高并发）。
"""

import threading
import time


class TokenBucket:
    """单实例令牌桶，acquire() 阻塞直到拿到令牌。"""

    def __init__(self, rpm_limit: int, burst: int | None = None):
        if rpm_limit <= 0:
            raise ValueError(f"rpm_limit 必须 > 0，当前 {rpm_limit}")
        self.capacity = burst if burst is not None else rpm_limit
        self.refill_rate = rpm_limit / 60.0  # 每秒补充的令牌数
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last = now

    def acquire(self, n: int = 1) -> None:
        """阻塞直到拿到 n 个令牌。"""
        if n > self.capacity:
            raise ValueError(f"请求令牌数 {n} 超过桶容量 {self.capacity}")
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                wait_seconds = deficit / self.refill_rate
            time.sleep(wait_seconds)


# 模块级单例缓存：每个 (rpm_limit, burst) 组合共享一个桶
_buckets: dict[tuple, TokenBucket] = {}
_lock = threading.Lock()


def get_bucket(rpm_limit: int, burst: int | None = None) -> TokenBucket:
    """获取（或创建）共享的令牌桶实例。"""
    key = (rpm_limit, burst)
    with _lock:
        if key not in _buckets:
            _buckets[key] = TokenBucket(rpm_limit, burst)
        return _buckets[key]