"""
merge_outputs.py - 合并各 checkpoint 输出最终结果

合并规则（以 id 为主键）：
1. 基础数据来自 02_filtered.jsonl
2. 用 03_extracted.jsonl 覆盖 metadata 和 density_score
3. 用 04_hyde.jsonl 追加 hyde 字段
4. 输出 05_final_output.json（含全量）+ 05_final_output.jsonl（流式版）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import file_utils
from common.config_loader import (
    CHECKPOINT_DIR, OUTPUT_DIR, ModelConfig, ThemeConfig,
    load_models, load_theme,
)
from common.logger import get_logger

logger = get_logger("merge")

FILTERED_PATH = CHECKPOINT_DIR / "02_filtered.jsonl"
EXTRACTED_PATH = CHECKPOINT_DIR / "03_extracted.jsonl"
HYDE_PATH = CHECKPOINT_DIR / "04_hyde.jsonl"
JSON_OUTPUT = OUTPUT_DIR / "05_final_output.json"
JSONL_OUTPUT = OUTPUT_DIR / "05_final_output.jsonl"


def _index_by_id(records, key="id"):
    return {r[key]: r for r in records if key in r}


def run(theme: ThemeConfig | None = None,
        models: ModelConfig | None = None) -> None:
    theme = theme or load_theme()
    load_models()  # 校验配置可加载

    logger.info("merge 启动")
    filtered = list(file_utils.read_jsonl(FILTERED_PATH))
    extracted = _index_by_id(file_utils.read_jsonl(EXTRACTED_PATH))
    hyde = _index_by_id(file_utils.read_jsonl(HYDE_PATH))

    final: list[dict] = []
    dropped_unrelated = 0
    for rec in filtered:
        cid = rec["id"]
        relevance = rec.get("relevance")
        # unrelated 块直接过滤掉，不参与后续 embedding
        if relevance == "unrelated":
            dropped_unrelated += 1
            continue
        out = {
            "id": cid,
            "text": rec.get("text", ""),
            "relevance": relevance,
            "rough_density": rec.get("rough_density"),
            "density_score": rec.get("density_score", 0.0),
            "metadata": {},
            "hyde": None,
            "model_versions": {
                "theme": theme.name,
            },
        }
        if cid in extracted:
            ext = extracted[cid]
            out["metadata"] = ext.get("metadata", {})
            out["density_score"] = ext.get("density_score", out["density_score"])
        if cid in hyde:
            out["hyde"] = hyde[cid].get("hyde")
        final.append(out)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_utils.write_json(JSON_OUTPUT, final)
    file_utils.write_jsonl(JSONL_OUTPUT, final)
    logger.info("merge 完成: 输入 %d，过滤 unrelated %d，最终 %d → %s, %s",
                len(filtered), dropped_unrelated, len(final), JSON_OUTPUT, JSONL_OUTPUT)


if __name__ == "__main__":
    run()