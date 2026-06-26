"""
config_loader.py - 配置加载与校验

支持：
- 主题配置（JSON）：业务维度、阈值
- 模型配置（YAML）：模型类型、限流、批次大小
- Prompt 模板（Markdown）：动态注入主题名等变量
- CLI > 环境变量 > 默认值 优先级
- 每个输入文件对应一个 run_dir，所有产物集中存放

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
# 兼容旧代码：仍导出 CHECKPOINT_DIR / OUTPUT_DIR，但默认产物改落到 run_dir
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


@dataclass
class RunPaths:
    """单个 input 文件对应的运行目录所有路径。"""
    run_dir: Path
    filtered: Path
    filtered_pending: Path
    extracted: Path
    extracted_pending: Path
    hyde: Path
    hyde_pending: Path
    final_json: Path
    final_jsonl: Path

    @classmethod
    def for_run_dir(cls, run_dir: str | Path) -> "RunPaths":
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            run_dir=run_dir,
            filtered=run_dir / "02_filtered.jsonl",
            filtered_pending=run_dir / "02_pending.jsonl",
            extracted=run_dir / "03_extracted.jsonl",
            extracted_pending=run_dir / "03_pending.jsonl",
            hyde=run_dir / "04_hyde.jsonl",
            hyde_pending=run_dir / "04_pending.jsonl",
            final_json=run_dir / "05_final_output.json",
            final_jsonl=run_dir / "05_final_output.jsonl",
        )


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


def build_chunks_prompt(template_name: str, theme: ThemeConfig,
                        chunks: list[dict], **vars) -> str:
    """加载模板、注入主题名+额外变量、追加 [id]\\ntext 形式的 chunks 块。"""
    template = load_prompt(template_name, theme=theme.name, **vars)
    blocks = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in chunks)
    return f"{template}\n\n---\n\n{blocks}"


# ---------------------------------------------------------------------------
# CLI / 环境变量辅助
# ---------------------------------------------------------------------------
def resolve_input_path(path: str | Path | None = None) -> Path:
    p = Path(path or os.environ.get("KB_INPUT_FILE") or INPUT_DIR / "01_raw_chunks.json")
    if not p.exists():
        raise FileNotFoundError(f"输入文件不存在: {p}")
    return p


# data/ 下作为运行目录的子目录（不是中间产物）
_RUN_DIR_RESERVED = {"input", "checkpoints", "output", "logs"}


def derive_run_dir(input_path: str | Path) -> Path:
    """从任意输入路径推导 run_dir。

    规则：
    - 路径本身就是 data/<X>/<file>：若 X 不是保留目录，run_dir = data/<X>/
      （例如 data/foo/02_filtered.jsonl → data/foo/）
    - 路径在 data/<X>/file 但 X 是保留目录：run_dir = data/<stem>/
      （例如 data/input/foo.json → data/foo/）
    - 其它位置：run_dir = data/<stem>/
    """
    p = Path(input_path).resolve()
    if p.is_dir():
        return p

    parent = p.parent
    try:
        rel_parent = parent.relative_to(DATA_DIR.resolve())
    except ValueError:
        return DATA_DIR / p.stem

    if len(rel_parent.parts) == 1 and rel_parent.parts[0] not in _RUN_DIR_RESERVED:
        return parent
    return DATA_DIR / p.stem