"""
model_factory.py - 模型工厂（按配置实例化）

业务模块只调用 create_model(entry)，不直接 import 具体实现。
添加新模型：在 _REGISTRY 中注册一个 (type, class) 对即可。
"""

from typing import Any

from common.model_interface import ModelEntry, ModelInterface
from common.model_local_qwen import LocalQwenModel
from common.model_minimax import MinimaxModel

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