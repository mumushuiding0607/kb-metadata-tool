"""
extractor.py - 第二步：高级模型精炼

输入：02_filtered.jsonl 中 density_score < 0 的块（待精炼）
输出：
  - 03_extracted.jsonl：成功批次（含元数据 + density_score）
  - 03_pending.jsonl：失败批次（待下次重试）

工程参数：
- batch_size = 5
- 无 max_tokens（用 timeout=30 控制）
- 重试 2 次：5s、10s 指数退避（通过 common.retry.with_retry）
- 失败入 pending，不污染主数据
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import file_utils
from common.batch_runner import run_pipeline
from common.config_loader import (
    CHECKPOINT_DIR, ModelConfig, ThemeConfig, load_models, load_theme,
)
from common.logger import get_logger
from common.model_factory import create_model
from common.model_interface import TransientError
from common.retry import with_retry
from modules.extractor_prompt import build_extract_prompt, parse_extract_response

logger = get_logger("extractor")

EXTRACTED_PATH = CHECKPOINT_DIR / "03_extracted.jsonl"
PENDING_PATH = CHECKPOINT_DIR / "03_pending.jsonl"
INPUT_PATH = CHECKPOINT_DIR / "02_filtered.jsonl"


def _process_batch(model, theme: ThemeConfig, batch: list[dict]) -> dict[str, dict]:
    """处理单个批次，返回 {id: result_dict}；失败抛 TransientError。"""
    ids = [c["id"] for c in batch]
    prompt = build_extract_prompt(theme, batch)

    def _do_call():
        resp = model.generate(prompt, timeout=30)
        aligned = parse_extract_response(resp.text, ids)
        if not aligned:
            raise TransientError("解析结果为空")
        return aligned

    return with_retry(_do_call, description=f"extractor batch(ids={ids[:2]}...)")


def _load_pending_chunks() -> list[dict]:
    """从 02_filtered.jsonl 加载 density_score < 0 的块（待精炼）。"""
    return [r for r in file_utils.read_jsonl(INPUT_PATH)
            if r.get("density_score", -1) < 0]


def run(theme: ThemeConfig | None = None,
        models: ModelConfig | None = None) -> None:
    theme = theme or load_theme()
    models = models or load_models()

    logger.info("extractor 启动: theme=%s", theme.name)
    model = create_model(models.extractor_model)

    chunks = _load_pending_chunks()
    dim_keys = [d["key"] for d in theme.dimensions]

    def _build_success(chunk: dict, result: dict) -> dict:
        return {
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": {k: result.get(k, "") for k in dim_keys},
            "density_score": float(result.get("density_score", 0.5)),
        }

    def _build_pending(chunk: dict, reason: str) -> dict:
        return {"id": chunk["id"], "text": chunk["text"], "reason": reason}

    def _process_fn(batch: list[dict]) -> dict[str, dict]:
        return _process_batch(model, theme, batch)

    run_pipeline(
        chunks,
        process_batch=_process_fn,
        success_path=EXTRACTED_PATH,
        pending_path=PENDING_PATH,
        build_success=_build_success,
        build_pending=_build_pending,
        batch_size=models.extractor_model.batch_size,
        description="extractor",
    )


if __name__ == "__main__":
    run()