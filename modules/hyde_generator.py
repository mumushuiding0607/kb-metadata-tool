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
from common.config_loader import (
    CHECKPOINT_DIR, ModelConfig, ThemeConfig,
    load_models, load_prompt, load_theme,
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
    template = load_prompt(
        "step3_hyde",
        theme=theme.name,
        max_chars=theme.hyde_max_chars,
    )
    blocks = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in chunks)
    return f"{template}\n\n---\n\n{blocks}"


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
    records = list(file_utils.read_jsonl(EXTRACTED_PATH))
    return [r for r in records if r.get("density_score", 0) >= theme.density_threshold]


def run(theme: ThemeConfig | None = None,
        models: ModelConfig | None = None) -> None:
    theme = theme or load_theme()
    models = models or load_models()

    logger.info("hyde 启动: theme=%s, threshold=%.2f",
                theme.name, theme.density_threshold)
    model = create_model(models.hyde_model)

    chunks = _load_qualified(theme)
    logger.info("高质量块数: %d", len(chunks))
    if not chunks:
        return

    completed = file_utils.read_completed_ids(HYDE_PATH) \
        | file_utils.read_completed_ids(HYDE_PENDING_PATH)
    pending = [c for c in chunks if c["id"] not in completed]
    logger.info("已完成 %d，待处理 %d", len(completed), len(pending))
    if not pending:
        return

    batch_size = models.hyde_model.batch_size
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        ids = [c["id"] for c in batch]
        try:
            results = _process_batch(model, theme, batch)
        except Exception as e:
            logger.error("hyde 批次失败 ids=%s: %s", ids, e)
            for c in batch:
                file_utils.append_jsonl(HYDE_PENDING_PATH, {
                    "id": c["id"], "text": c["text"], "reason": str(e),
                })
            continue
        for c in batch:
            q = results.get(c["id"], "")
            if not q:
                file_utils.append_jsonl(HYDE_PENDING_PATH, {
                    "id": c["id"], "text": c["text"], "reason": "missing",
                })
                continue
            file_utils.append_jsonl(HYDE_PATH, {
                "id": c["id"],
                "hyde": q,
                "density_score": c.get("density_score", 0),
            })
        logger.info("hyde 进度 %d/%d", min(i + batch_size, len(pending)), len(pending))

    logger.info("hyde 完成")


if __name__ == "__main__":
    run()