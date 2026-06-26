"""
filter.py - 第一步：本地模型粗筛（relevance + rough_density）

输入：01_raw_chunks.json
输出：02_filtered.jsonl（追加写，支持断点续传）

分流规则：
- relevance == 'unrelated'       → density_score = 0.0，永久归档
- rough_density == 'low'（相关） → density_score = 0.3，不送第二步
- 其余有效块                     → density_score = -1（占位，第二步计算）
"""

import sys
import time
from pathlib import Path

# 允许从项目根目录直接 python modules/filter.py 运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import file_utils
from common.config_loader import (
    CHECKPOINT_DIR, ModelConfig, ThemeConfig,
    build_chunks_prompt, load_models, load_theme, resolve_input_path,
)
from common.logger import get_logger
from common.model_factory import create_model

logger = get_logger("filter")


# ---------------------------------------------------------------------------
# 输出路径
# ---------------------------------------------------------------------------
OUTPUT_PATH = CHECKPOINT_DIR / "02_filtered.jsonl"


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------
def _build_prompt(theme: ThemeConfig, chunks: list[dict]) -> str:
    return build_chunks_prompt("step1_filter", theme, chunks)


# ---------------------------------------------------------------------------
# 解析模型输出（按 ID 提取标签）
# ---------------------------------------------------------------------------
def _parse_labels(raw: str, chunk_ids: list[str]) -> dict[str, dict]:
    """从模型输出中按 id 提取 (relevance, rough_density)。"""
    result: dict[str, dict] = {}
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for line in lines:
        for cid in chunk_ids:
            if cid in line:
                rel = "unrelated"
                rd = "low"
                low = line.lower()
                if "direct" in low:
                    rel = "direct"
                elif "inspirational" in low:
                    rel = "inspirational"
                if "high" in low:
                    rd = "high"
                elif "medium" in low:
                    rd = "medium"
                result[cid] = {"relevance": rel, "rough_density": rd}
                break
    # 缺失的块默认标记为 unrelated / low（保守）
    for cid in chunk_ids:
        if cid not in result:
            result[cid] = {"relevance": "unrelated", "rough_density": "low"}
    return result


def _to_density_score(labels: dict) -> float:
    rel = labels["relevance"]
    rd = labels["rough_density"]
    if rel == "unrelated":
        return 0.0
    if rd == "low":
        return 0.3
    return -1.0  # 送第二步精炼


# ---------------------------------------------------------------------------
# 主编排
# ---------------------------------------------------------------------------
def run(theme: ThemeConfig | None = None,
        models: ModelConfig | None = None,
        input_path: str | Path | None = None) -> None:
    theme = theme or load_theme()
    models = models or load_models()
    input_path = resolve_input_path(input_path)

    logger.info("filter 启动: theme=%s, input=%s", theme.name, input_path)
    model = create_model(models.filter_model)

    chunks = list(file_utils.read_jsonl(input_path))
    if not chunks:
        # 也支持单 JSON 数组
        chunks = file_utils.read_json(input_path) or []
    logger.info("待处理块数: %d", len(chunks))

    completed = file_utils.read_completed_ids(OUTPUT_PATH)
    pending = [c for c in chunks if c["id"] not in completed]
    logger.info("已完成 %d，跳过；待处理 %d", len(completed), len(pending))

    if not pending:
        logger.info("filter 无需处理，退出")
        return

    batch_size = models.filter_model.batch_size
    start = time.monotonic()
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        ids = [c["id"] for c in batch]
        prompt = _build_prompt(theme, batch)
        try:
            resp = model.generate(prompt, timeout=models.filter_model.timeout)
            labels_map = _parse_labels(resp.text, ids)
        except Exception as e:
            logger.error("filter 批次失败 ids=%s: %s", ids[:3], e)
            labels_map = {cid: {"relevance": "unrelated", "rough_density": "low"} for cid in ids}

        for chunk in batch:
            labels = labels_map.get(chunk["id"], {"relevance": "unrelated", "rough_density": "low"})
            score = _to_density_score(labels)
            file_utils.append_jsonl(OUTPUT_PATH, {
                "id": chunk["id"],
                "text": chunk["text"],
                "relevance": labels["relevance"],
                "rough_density": labels["rough_density"],
                "density_score": score,
            })
        logger.info("filter 进度 %d/%d", min(i + batch_size, len(pending)), len(pending))

    logger.info("filter 完成，耗时 %.1fs", time.monotonic() - start)


if __name__ == "__main__":
    run()