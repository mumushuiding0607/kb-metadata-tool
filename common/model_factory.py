"""
model_factory.py - 模型工厂（按配置实例化）

支持的 type：
- minimax_pro: 调用 Minimax API（Anthropic 协议）
- local_qwen:  本地 Qwen 模型（transformers 或 vllm，按 extra.framework 选择）

业务模块只调用 create_model(entry)，不直接 import 具体实现。
"""

import os
import re
import time
from typing import Any

import requests

from common.logger import get_logger
from common.model_interface import (
    ModelEntry, ModelInterface, ModelResponse,
    PermanentError, TransientError,
)
from common.token_bucket import get_bucket

logger = get_logger("model_factory")


# ---------------------------------------------------------------------------
# Minimax 云端模型
# ---------------------------------------------------------------------------
class MinimaxModel:
    """调用 Minimax API（Anthropic Messages 协议）。"""

    _THINKING_TRUNC = 500

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
        start = time.monotonic()
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=self.timeout)
        except requests.Timeout as e:
            raise TransientError(f"timeout after {self.timeout}s") from e
        except requests.ConnectionError as e:
            raise TransientError(f"connection error: {e}") from e

        latency = int((time.monotonic() - start) * 1000)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise TransientError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise PermanentError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return _parse_anthropic_response(resp.json(), latency)

    def generate(self, prompt: str, timeout: int = 30) -> ModelResponse:
        return self._call(prompt)

    def batch_generate(self, prompts: list[str], timeout: int = 30) -> list[ModelResponse]:
        # 串行执行，限流已在 _call 内部处理
        return [self._call(p) for p in prompts]


def _parse_anthropic_response(data: dict, latency: int) -> ModelResponse:
    """解析 Anthropic Messages 响应，提取 text/thinking 并合并。"""
    blocks = data.get("content", [])
    text_parts: list[str] = []
    for block in blocks:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            text_parts.append(block.get("thinking", "")[:MinimaxModel._THINKING_TRUNC])
    usage = data.get("usage", {})
    return ModelResponse(
        text="\n".join(text_parts),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# 本地 Qwen 模型（CPU/GPU，按 framework 选择）
# ---------------------------------------------------------------------------
class LocalQwenModel:
    """本地 Qwen 模型。支持 transformers 和 vllm 两种 backend。"""

    def __init__(self, entry: ModelEntry):
        self.model_path = entry.extra.get("model_path", "")
        self.device = entry.extra.get("device", "cpu")
        self.framework = entry.extra.get("framework", "transformers")
        self.batch_size = entry.batch_size
        if not self.model_path:
            raise RuntimeError("local_qwen 模型需要 extra.model_path")
        self._model = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        if self.framework == "vllm":
            from vllm import LLM
            self._model = LLM(model=self.model_path, dtype="float16")
            logger.info("vllm 模型加载完成: %s", self.model_path)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=dtype
            ).to(self.device)
            logger.info("transformers 模型加载完成: %s (device=%s)",
                        self.model_path, self.device)

    def _infer(self, prompts: list[str]) -> list[str]:
        if self.framework == "vllm":
            outputs = self._model.generate(prompts, use_tqdm=False)
            return [o.outputs[0].text for o in outputs]
        results: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            inputs = self._tokenizer(batch, return_tensors="pt", padding=True).to(self.device)
            with __import__("torch").no_grad():
                out = self._model.generate(**inputs, max_new_tokens=512)
            for j, prompt in enumerate(batch):
                gen = out[j][inputs["input_ids"].shape[1]:]
                results.append(self._tokenizer.decode(gen, skip_special_tokens=True))
        return results

    def generate(self, prompt: str, timeout: int = 30) -> ModelResponse:
        start = time.monotonic()
        text = self._infer([prompt])[0]
        return ModelResponse(text=text, latency_ms=int((time.monotonic() - start) * 1000))

    def batch_generate(self, prompts: list[str], timeout: int = 30) -> list[ModelResponse]:
        start = time.monotonic()
        texts = self._infer(prompts)
        latency = int((time.monotonic() - start) * 1000)
        return [ModelResponse(text=t, latency_ms=latency) for t in texts]


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Any] = {
    "minimax_pro": MinimaxModel,
    "local_qwen": LocalQwenModel,
}


def create_model(entry: ModelEntry) -> ModelInterface:
    """按 entry.type 实例化模型。"""
    cls = _REGISTRY.get(entry.type)
    if cls is None:
        raise ValueError(f"未知模型类型: {entry.type}，已注册: {list(_REGISTRY)}")
    return cls(entry)