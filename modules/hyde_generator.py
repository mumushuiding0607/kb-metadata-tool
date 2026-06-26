"""
hyde_generator.py - 第三步：HyDE 生成（≤theme.hyde_max_chars 字）

输入：03_extracted.jsonl 中 density_score >= theme.density_threshold 的块
输出：
  - 04_hyde.jsonl：成功
  - 04_pending.jsonl：失败
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import file_utils
from common.batch_runner import run_pipeline
from common.config_loader import (
    CHECKPOINT_DIR, ModelConfig, ThemeConfig,
    build_chunks_prompt, load_models, load_theme,
)
from common.logger import get_logger
from common.model_factory import create_model
from common.model_interface import TransientError
from common.retry import with_retry

logger = get_logger("hyde")

HYDE_PATH = CHECKPOINT_DIR / "04_hyde.jsonl"
HYDE_PENDING_PATH = CHECKPOINT_DIR / "04_pending.jsonl"
EXTRACTED_PATH = CHECKPOINT_DIR / "03_extracted.jsonl"


def _build_prompt(theme: ThemeConfig, chunks: list[dict]) -> str:
    return build_chunks_prompt(
        "step3_hyde", theme, chunks, max_chars=theme.hyde_max_chars,
    )


def _parse_hyde(raw: str, expected_ids: list[str]) -> dict[str, str]:
    """按 [id] question 格式解析。"""
    result: dict[str, str] = {}
    for cid in expected_ids:
        m = re.search(rf"\[{re.escape(cid)}\]\s*(.+?)(?=\n\[|$)", raw, re.DOTALL)
        if m:
            question = m.group(1).strip()
            if question:
                result[cid] = question
    return result


def _process_batch(model, theme: ThemeConfig, batch: list[dict]) -> dict[str, str]:
    ids = [c["id"] for c in batch]
    prompt = _build_prompt(theme, batch)

    def _do_call():
        resp = model.generate(prompt, timeout=30)
        parsed = _parse_hyde(resp.text, ids)
        if not parsed:
            raise TransientError("hyde 解析结果为空")
        return parsed

    return with_retry(_do_call, description=f"hyde batch(ids={ids[:2]}...)")


def _load_qualified(theme: ThemeConfig) -> list[dict]:
    return [r for r in file_utils.read_jsonl(EXTRACTED_PATH)
            if r.get("density_score", 0) >= theme.density_threshold]


def run(theme: ThemeConfig | None = None,
        models: ModelConfig | None = None) -> None:
    theme = theme or load_theme()
    models = models or load_models()

    logger.info("hyde 启动: theme=%s, threshold=%.2f",
                theme.name, theme.density_threshold)
    model = create_model(models.hyde_model)

    chunks = _load_qualified(theme)

    def _build_success(chunk: dict, result: dict) -> dict:
        return {
            "id": chunk["id"],
            "hyde": result if isinstance(result, str) else result.get("hyde", ""),
            "density_score": chunk.get("density_score", 0),
        }

    def _build_pending(chunk: dict, reason: str) -> dict:
        return {"id": chunk["id"], "text": chunk["text"], "reason": reason}

    def _process_fn(batch: list[dict]) -> dict[str, str]:
        return _process_batch(model, theme, batch)

    run_pipeline(
        chunks,
        process_batch=_process_fn,
        success_path=HYDE_PATH,
        pending_path=HYDE_PENDING_PATH,
        build_success=_build_success,
        build_pending=_build_pending,
        batch_size=models.hyde_model.batch_size,
        description="hyde",
    )


if __name__ == "__main__":
    run()