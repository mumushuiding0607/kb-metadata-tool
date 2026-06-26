"""
config_loader.py - 配置加载与校验

支持：
- 主题配置（JSON）：业务维度、阈值
- 模型配置（YAML）：模型类型、限流、批次大小
- Prompt 模板（Markdown）：动态注入主题名等变量
- CLI > 环境变量 > 默认值 优先级

业务模块只能通过 load_*() 函数获取配置，不自行解析文件。
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from common.model_interface import ModelEntry

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
OUTPUT_DIR = DATA_DIR / "output"


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------
@dataclass
class ThemeConfig:
    name: str
    dimensions: list[dict]
    relevance_labels: list[str] = field(default_factory=lambda: ["direct", "inspirational", "unrelated"])
    density_threshold: float = 0.7
    hyde_max_chars: int = 50


@dataclass
class ModelConfig:
    filter_model: ModelEntry
    extractor_model: ModelEntry
    hyde_model: ModelEntry


# ---------------------------------------------------------------------------
# 加载函数
# ---------------------------------------------------------------------------
def load_theme(path: str | Path | None = None) -> ThemeConfig:
    """加载主题配置，路径可通过 CLI/env 指定，默认 ai_monetization。"""
    p = Path(path or os.environ.get("KB_THEME_CONFIG")
             or CONFIG_DIR / "theme" / "ai_monetization.json")
    if not p.exists():
        raise FileNotFoundError(f"主题配置不存在: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))

    # 基础校验
    for key in ("theme_name", "dimensions"):
        if key not in raw:
            raise ValueError(f"主题配置 {p} 缺少必需字段: {key}")

    threshold = float(raw.get("density_threshold", 0.7))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"density_threshold 必须在 [0,1]，当前 {threshold}")

    hyde = raw.get("hyde_constraints", {})
    return ThemeConfig(
        name=raw["theme_name"],
        dimensions=raw["dimensions"],
        relevance_labels=raw.get("relevance_labels", ["direct", "inspirational", "unrelated"]),
        density_threshold=threshold,
        hyde_max_chars=int(hyde.get("max_chars", 50)),
    )


def _parse_model_entry(d: dict, default_timeout: int = 30) -> ModelEntry:
    return ModelEntry(
        type=d["type"],
        rpm_limit=int(d.get("rpm_limit", 0)),
        batch_size=int(d.get("batch_size", 1)),
        timeout=int(d.get("timeout_seconds", default_timeout)),
        extra={k: v for k, v in d.items()
               if k not in ("type", "rpm_limit", "batch_size", "timeout_seconds")},
    )


def load_models(path: str | Path | None = None) -> ModelConfig:
    """加载模型路由配置。"""
    p = Path(path or os.environ.get("KB_MODEL_CONFIG")
             or CONFIG_DIR / "models" / "local_minimax.yaml")
    if not p.exists():
        raise FileNotFoundError(f"模型配置不存在: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))

    for key in ("filter_model", "extractor_model", "hyde_model"):
        if key not in raw:
            raise ValueError(f"模型配置 {p} 缺少必需字段: {key}")

    return ModelConfig(
        filter_model=_parse_model_entry(raw["filter_model"]),
        extractor_model=_parse_model_entry(raw["extractor_model"]),
        hyde_model=_parse_model_entry(raw["hyde_model"]),
    )


def load_prompt(name: str, **vars) -> str:
    """加载 Prompt 模板，注入变量。name 不含 .md 后缀。"""
    p = CONFIG_DIR / "prompts" / f"{name}.md"
    if not p.exists():
        raise FileNotFoundError(f"Prompt 模板不存在: {p}")
    text = p.read_text(encoding="utf-8")
    for k, v in vars.items():
        text = text.replace("{" + k + "}", str(v))
    return text


# ---------------------------------------------------------------------------
# CLI / 环境变量辅助
# ---------------------------------------------------------------------------
def resolve_input_path(path: str | Path | None = None) -> Path:
    p = Path(path or os.environ.get("KB_INPUT_FILE") or INPUT_DIR / "01_raw_chunks.json")
    if not p.exists():
        raise FileNotFoundError(f"输入文件不存在: {p}")
    return p