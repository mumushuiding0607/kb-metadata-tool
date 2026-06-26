"""
filter.py - 第一步：本地模型粗筛

输入：原始分块文件（JSON 或 JSONL）
输出：run_dir/02_filtered.jsonl（已剔除 unrelated），run_dir/02_pending.jsonl（失败批次）

filter 之后，unrelated 块直接丢弃，不再参与后续处理。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import file_utils
from common.batch_runner import run_pipeline
from common.config_loader import (
    ModelConfig, RunPaths, ThemeConfig, build_chunks_prompt,
)
from common.logger import get_logger
from common.model_factory import create_model
from common.retry import with_retry
from common.run_context import setup_run

logger = get_logger("filter")


def _build_prompt(theme: ThemeConfig, chunks: list[dict]) -> str:
    return build_chunks_prompt("step1_filter", theme, chunks)


def _parse_labels(raw: str, chunk_ids: list[str]) -> dict[str, dict]:
    """从模型输出中按 id 提取 (relevance, rough_density)。"""
    result: dict[str, dict] = {}
    for line in (ln.strip() for ln in raw.splitlines() if ln.strip()):
        for cid in chunk_ids:
            if cid in line:
                low = line.lower()
                rel = "direct" if "direct" in low \
                    else "inspirational" if "inspirational" in low else "unrelated"
                rd = "high" if "high" in low \
                    else "medium" if "medium" in low else "low"
                result[cid] = {"relevance": rel, "rough_density": rd}
                break
    # 缺失的块保守标记为 unrelated（将被丢弃）
    for cid in chunk_ids:
        result.setdefault(cid, {"relevance": "unrelated", "rough_density": "low"})
    return result


def _process_batch(model, theme: ThemeConfig, batch: list[dict]) -> dict[str, dict]:
    ids = [c["id"] for c in batch]
    prompt = _build_prompt(theme, batch)

    def _do_call():
        resp = model.generate(prompt, timeout=model.timeout if hasattr(model, "timeout") else 30)
        return _parse_labels(resp.text, ids)

    return with_retry(_do_call, description=f"filter batch(ids={ids[:2]}...)")


def _load_input_chunks(input_path: Path) -> list[dict]:
    chunks = list(file_utils.read_jsonl(input_path))
    if not chunks:
        chunks = file_utils.read_json(input_path) or []
    return chunks


def run(theme: ThemeConfig | None = None,
        models: ModelConfig | None = None,
        input_path: str | Path | None = None,
        run_dir: str | Path | None = None) -> RunPaths:
    ctx = setup_run("filter", theme=theme, models=models,
                    input_path=input_path, run_dir=run_dir)
    if input_path is not None:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

    logger.info("filter 启动: theme=%s, run_dir=%s", ctx.theme.name, ctx.paths.run_dir)
    if input_path is None:
        # 仅给 run_dir 模式（重试场景）：从 02_pending 恢复
        pending_chunks = list(file_utils.read_jsonl(ctx.paths.filtered_pending))
        if not pending_chunks:
            logger.info("filter 无 pending，退出")
            return ctx.paths
        chunks = pending_chunks
    else:
        chunks = _load_input_chunks(input_path)
    logger.info("待处理块数: %d", len(chunks))

    model = create_model(ctx.models.filter_model)

    def _build_success(chunk: dict, labels: dict) -> dict | None:
        # unrelated 直接丢弃（filter 的核心职责）
        if labels["relevance"] == "unrelated":
            return None
        return {
            "id": chunk["id"],
            "text": chunk["text"],
            "relevance": labels["relevance"],
            "rough_density": labels["rough_density"],
        }

    def _build_pending(chunk: dict, reason: str) -> dict:
        return {"id": chunk["id"], "text": chunk["text"], "reason": reason}

    def _process_fn(batch: list[dict]) -> dict[str, dict]:
        return _process_batch(model, ctx.theme, batch)

    success, fail = run_pipeline(
        chunks,
        process_batch=_process_fn,
        success_path=ctx.paths.filtered,
        pending_path=ctx.paths.filtered_pending,
        build_success=_build_success,
        build_pending=_build_pending,
        batch_size=ctx.models.filter_model.batch_size,
        description="filter",
    )
    logger.info("filter 完成: 通过 %d，丢弃/失败 %d", success, fail)
    return ctx.paths


if __name__ == "__main__":
    run()
