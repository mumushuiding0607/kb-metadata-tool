"""
merge_outputs.py - 合并 run_dir 内的 checkpoint 输出最终结果

合并规则（以 id 为主键）：
1. 基础数据来自 02_filtered.jsonl（filter 已剔除 unrelated，无需再次过滤）
2. 用 03_extracted.jsonl 覆盖 metadata 和 density_score
3. 用 04_hyde.jsonl 追加 hyde 字段
4. 输出 05_final_output.json + 05_final_output.jsonl 到 run_dir
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import file_utils
from common.config_loader import (
    ModelConfig, RunPaths, ThemeConfig, derive_run_dir, load_theme,
)
from common.logger import get_logger

logger = get_logger("merge")


def _index_by_id(records) -> dict:
    return {r["id"]: r for r in records if "id" in r}


def run(theme: ThemeConfig | None = None,
        models: ModelConfig | None = None,
        input_path: str | Path | None = None,
        run_dir: str | Path | None = None) -> RunPaths:
    theme = theme or load_theme()

    if run_dir is None:
        if input_path is None:
            raise ValueError("merge 需要 input_path 或 run_dir")
        run_dir = derive_run_dir(input_path)
    paths = RunPaths.for_run_dir(run_dir)

    logger.info("merge 启动: run_dir=%s", paths.run_dir)
    extracted = _index_by_id(file_utils.read_jsonl(paths.extracted))
    hyde = _index_by_id(file_utils.read_jsonl(paths.hyde))
    filtered = list(file_utils.read_jsonl(paths.filtered))

    final: list[dict] = []
    for rec in filtered:
        cid = rec["id"]
        out = {
            "id": cid,
            "text": rec.get("text", ""),
            "relevance": rec.get("relevance"),
            "rough_density": rec.get("rough_density"),
            "density_score": 0.0,
            "metadata": {},
            "hyde": None,
            "model_versions": {"theme": theme.name},
        }
        if cid in extracted:
            ext = extracted[cid]
            out["metadata"] = ext.get("metadata", {})
            out["density_score"] = ext.get("density_score", 0.0)
        if cid in hyde:
            out["hyde"] = hyde[cid].get("hyde")
        final.append(out)

    file_utils.write_json(paths.final_json, final)
    file_utils.write_jsonl(paths.final_jsonl, final)
    logger.info("merge 完成: 输入 %d → %s, %s",
                len(final), paths.final_json, paths.final_jsonl)
    return paths


if __name__ == "__main__":
    run()