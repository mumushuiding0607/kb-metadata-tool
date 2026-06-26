"""
extractor.py - 第二步：高级模型精炼（最核心、最易崩溃的环节）

输入：02_filtered.jsonl 中 density_score < 0 的块（待精炼）
输出：
  - 03_extracted.jsonl：成功批次（含元数据 + density_score）
  - 03_pending.jsonl：失败批次（待下次重试）

工程参数：
- batch_size = 5
- 无 max_tokens（用 timeout=30 控制）
- 重试 2 次：5s、10s 指数退避（通过 common.retry.with_retry）
- 失败入 pending，不污染主数据
- 每批成功立即追加写
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import file_utils
from common.config_loader import (
    CHECKPOINT_DIR, ModelConfig, ThemeConfig,
    load_models, load_theme,
)
from common.logger import get_logger
from common.model_factory import create_model
from common.retry import with_retry
from modules.extractor_prompt import build_extract_prompt, parse_extract_response

logger = get_logger("extractor")

EXTRACTED_PATH = CHECKPOINT_DIR / "03_extracted.jsonl"
PENDING_PATH = CHECKPOINT_DIR / "03_pending.jsonl"
INPUT_PATH = CHECKPOINT_DIR / "02_filtered.jsonl"


def _process_batch(model, theme: ThemeConfig, batch: list[dict]) -> dict[str, dict]:
    """处理单个批次，返回 {id: result_dict}；失败抛 TransientError。"""
    ids = [c["id"] for c in batch]
    prompt = build_extract_prompt(theme, batch, "")

    def _do_call():
        resp = model.generate(prompt, timeout=30)
        aligned = parse_extract_response(resp.text, ids)
        if not aligned:
            raise Exception("解析结果为空")
        return aligned

    return with_retry(_do_call, description=f"extractor batch(ids={ids[:2]}...)")


def _load_pending_chunks() -> list[dict]:
    """从 02_filtered.jsonl 加载 density_score < 0 的块（待精炼）。"""
    all_records = list(file_utils.read_jsonl(INPUT_PATH))
    return [r for r in all_records if r.get("density_score", -1) < 0]


def run(theme: ThemeConfig | None = None,
        models: ModelConfig | None = None) -> None:
    theme = theme or load_theme()
    models = models or load_models()

    logger.info("extractor 启动: theme=%s", theme.name)
    model = create_model(models.extractor_model)

    chunks = _load_pending_chunks()
    logger.info("待精炼块数: %d", len(chunks))
    if not chunks:
        logger.info("extractor 无需处理，退出")
        return

    completed = file_utils.read_completed_ids(EXTRACTED_PATH) \
        | file_utils.read_completed_ids(PENDING_PATH)
    pending = [c for c in chunks if c["id"] not in completed]
    logger.info("已完成 %d，跳过；待处理 %d", len(completed), len(pending))
    if not pending:
        return

    batch_size = models.extractor_model.batch_size
    dim_keys = [d["key"] for d in theme.dimensions]
    start = time.monotonic()
    success_count = fail_count = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        ids = [c["id"] for c in batch]
        try:
            results = _process_batch(model, theme, batch)
        except Exception as e:
            logger.error("批次最终失败 ids=%s: %s", ids, e)
            for c in batch:
                file_utils.append_jsonl(PENDING_PATH, {
                    "id": c["id"], "text": c["text"], "reason": str(e),
                })
            fail_count += len(batch)
            continue

        for c in batch:
            r = results.get(c["id"])
            if not r:
                file_utils.append_jsonl(PENDING_PATH, {
                    "id": c["id"], "text": c["text"], "reason": "missing",
                })
                fail_count += 1
                continue
            file_utils.append_jsonl(EXTRACTED_PATH, {
                "id": c["id"],
                "text": c["text"],
                "metadata": {k: r.get(k, "") for k in dim_keys},
                "density_score": float(r.get("density_score", 0.5)),
            })
            success_count += 1
        logger.info("extractor 进度 %d/%d（成功 %d，失败 %d）",
                    min(i + batch_size, len(pending)), len(pending),
                    success_count, fail_count)

    logger.info("extractor 完成，耗时 %.1fs，成功 %d，失败 %d",
                time.monotonic() - start, success_count, fail_count)


if __name__ == "__main__":
    run()