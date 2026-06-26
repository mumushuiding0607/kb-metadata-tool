"""
model_interface.py - 统一模型接口

所有模型实现都必须实现 ModelInterface 协议。
业务模块只通过接口交互，不依赖具体实现。
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ModelEntry:
    """模型配置项（从 YAML/JSON 解析得到）。"""
    type: str
    rpm_limit: int = 0
    batch_size: int = 1
    timeout: int = 30
    extra: dict = field(default_factory=dict)


@dataclass
class ModelResponse:
    """模型调用结果。"""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelInterface(Protocol):
    """模型接口协议——同步调用、批处理、释放资源。"""

    def generate(self, prompt: str, timeout: int = 30) -> ModelResponse:
        """单次调用，返回结构化结果。失败时抛 ModelError。"""
        ...

    def batch_generate(self, prompts: list[str], timeout: int = 30) -> list[ModelResponse]:
        """批量调用，逐个处理，返回与 prompts 等长的结果列表。"""
        ...


class ModelError(Exception):
    """模型调用错误基类。"""


class TransientError(ModelError):
    """可重试错误（超时、429、5xx）。"""


class PermanentError(ModelError):
    """不可重试错误（401、403、JSON 解析失败等）。"""