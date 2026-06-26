"""
model_minimax.py - Minimax 云端模型实现（Anthropic Messages 协议）

负责：
- 单次/并发 HTTP 调用
- 失败响应落盘到 data/logs/llm_errors/
- 详细日志（模型名、状态码、延迟、token 用量）
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from common.config_loader import DATA_DIR
from common.file_utils import append_jsonl
from common.logger import get_logger
from common.model_interface import (
    ModelEntry, ModelResponse, PermanentError, TransientError,
)
from common.token_bucket import get_bucket

logger = get_logger("minimax")
_THINKING_TRUNC = 500


class MinimaxModel:
    """调用 Minimax API（Anthropic Messages 协议）。"""

    def __init__(self, entry: ModelEntry):
        api_key = os.environ.get(entry.extra.get("api_key_env", "MINIMAX_API_KEY"), "")
        if not api_key:
            raise RuntimeError(
                f"模型 {entry.type} 需要环境变量 "
                f"{entry.extra.get('api_key_env', 'MINIMAX_API_KEY')}"
            )
        self.base_url = os.environ.get(
            "MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"
        )
        self.model_name = entry.extra.get(
            "model_name",
            os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7"),
        )
        self.api_key = api_key
        self.timeout = entry.timeout
        self.rpm_limit = entry.rpm_limit
        self._bucket = get_bucket(entry.rpm_limit) if entry.rpm_limit > 0 else None

    def _call(self, prompt: str) -> ModelResponse:
        if self._bucket:
            self._bucket.acquire()
        headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            # 不设 max_tokens，避免截断（用 timeout 控制）
        }
        url = f"{self.base_url}/v1/messages"
        prompt_chars = len(prompt)
        logger.info("llm_call_start model=%s prompt_chars=%d timeout=%ds",
                    self.model_name, prompt_chars, self.timeout)

        start = time.monotonic()
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=self.timeout)
        except requests.RequestException as e:
            latency = int((time.monotonic() - start) * 1000)
            if isinstance(e, requests.Timeout):
                logger.error("llm_call_timeout model=%s after=%dms timeout=%ds",
                             self.model_name, latency, self.timeout)
                raise TransientError(f"timeout after {self.timeout}s") from e
            logger.error("llm_call_request_error model=%s after=%dms err_type=%s err=%s",
                         self.model_name, latency, type(e).__name__, e)
            raise TransientError(f"{type(e).__name__}: {e}") from e

        latency = int((time.monotonic() - start) * 1000)
        if resp.status_code == 429 or resp.status_code >= 500:
            logger.warning("llm_call_server_error model=%s status=%d latency=%dms",
                           self.model_name, resp.status_code, latency)
            _save_error_response(resp, self.model_name, "server_error")
            raise TransientError(f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            logger.error("llm_call_client_error model=%s status=%d latency=%dms",
                         self.model_name, resp.status_code, latency)
            _save_error_response(resp, self.model_name, "client_error")
            raise PermanentError(f"HTTP {resp.status_code}")

        result = _parse_anthropic_response(resp.json(), latency)
        logger.info(
            "llm_call_ok model=%s status=%d latency=%dms "
            "in_tokens=%d out_tokens=%d resp_chars=%d",
            self.model_name, resp.status_code, latency,
            result.input_tokens, result.output_tokens, len(result.text),
        )
        return result

    def generate(self, prompt: str, timeout: int = 30) -> ModelResponse:
        return self._call(prompt)

    def batch_generate(self, prompts: list[str], timeout: int = 30) -> list[ModelResponse]:
        # 并发受 rpm_limit 约束：每秒最多 rpm_limit/60 个请求，因此 max_workers
        # 取 rpm_limit/12（小批量内允许瞬时并发，长时间由令牌桶节流）。
        if len(prompts) <= 1 or self.rpm_limit <= 0:
            return [self._call(p) for p in prompts]
        max_workers = max(1, min(len(prompts), self.rpm_limit // 12 or 1))
        # max_workers==1 时 ThreadPoolExecutor 是纯开销，直接顺序调用
        if max_workers == 1:
            logger.info("llm_batch_start model=%s n=%d sequential rpm_limit=%d",
                        self.model_name, len(prompts), self.rpm_limit)
            return [self._call(p) for p in prompts]
        logger.info("llm_batch_start model=%s n=%d max_workers=%d rpm_limit=%d",
                    self.model_name, len(prompts), max_workers, self.rpm_limit)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(self._call, prompts))
        logger.info("llm_batch_done model=%s n=%d",
                    self.model_name, len(prompts))
        return results


def _parse_anthropic_response(data: dict, latency: int) -> ModelResponse:
    """解析 Anthropic Messages 响应，提取 text/thinking 并合并。"""
    blocks = data.get("content", [])
    text_parts: list[str] = []
    for block in blocks:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            text_parts.append(block.get("thinking", "")[:_THINKING_TRUNC])
    usage = data.get("usage", {})
    return ModelResponse(
        text="\n".join(text_parts),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        latency_ms=latency,
    )


def _save_error_response(resp, model_name: str, category: str) -> None:
    """把失败响应的原始内容存到 data/logs/llm_errors/，便于事后排查。"""
    try:
        err_dir = DATA_DIR / "logs" / "llm_errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        append_jsonl(err_dir / f"{category}.jsonl", {
            "ts": time.time(),
            "model": model_name,
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "body": resp.text,
        })
    except Exception as e:
        logger.warning("save_error_response_failed: %s", e)