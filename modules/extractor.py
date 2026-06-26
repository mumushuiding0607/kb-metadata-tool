"""
extractor.py - 第二步：高级模型精炼

输入：run_dir/02_filtered.jsonl（filter 已剔除 unrelated）
输出：
  - run_dir/03_extracted.jsonl：成功批次
  - run_dir/03_pending.jsonl：失败批次

工程参数：batch_size=5、无 max_tokens、30s 超时、with_retry 5s/10s。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import file_utils
from common.batch_runner import run_pipeline
from common.config_loader import (
    ModelConfig, RunPaths, ThemeConfig,
    derive_run_dir, load_models, load_theme,
)
from common.logger import get_logger
from common.model_factory import create_model
from common.model_interface import TransientError
from common.retry import with_retry
from modules.extractor_prompt import build_extract_prompt, parse_extract_response

logger = get_logger("extractor")


def _process_batch(model, theme: ThemeConfig, batch: list[dict]) -> dict[str, dict]:
    ids = [c["id"] for c in batch]
    prompt = build_extract_prompt(theme, batch)

    def _do_call():
        resp = model.generate(prompt, timeout=30)
        aligned = parse_extract_response(resp.text, ids)
        if not aligned:
            raise TransientError("解析结果为空")
        return aligned

    return with_retry(_do_call, description=f"extractor batch(ids={ids[:2]}...)")


def _load_chunks(paths: RunPaths) -> list[dict]:
    return list(file_utils.read_jsonl(paths.filtered))


def run(theme: ThemeConfig | None = None,
        models: ModelConfig | None = None,
        input_path: str | Path | None = None,
        run_dir: str | Path | None = None) -> RunPaths:
    theme = theme or load_theme()
    models = models or load_models()

    if run_dir is None:
        if input_path is None:
            raise ValueError("extractor 需要 input_path 或 run_dir")
        run_dir = derive_run_dir(input_path)
    paths = RunPaths.for_run_dir(run_dir)

    if not paths.filtered.exists():
        raise FileNotFoundError(
            f"未找到 filter 输出: {paths.filtered}，请先执行 filter 步骤"
        )

    logger.info("extractor 启动: theme=%s, run_dir=%s", theme.name, paths.run_dir)
    chunks = _load_chunks(paths)
    logger.info("待精炼块数: %d", len(chunks))

    model = create_model(models.extractor_model)
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
        success_path=paths.extracted,
        pending_path=paths.extracted_pending,
        build_success=_build_success,
        build_pending=_build_pending,
        batch_size=models.extractor_model.batch_size,
        description="extractor",
    )
    return paths


if __name__ == "__main__":
    run()