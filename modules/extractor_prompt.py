"""
extractor_prompt.py - extractor 专属的 Prompt 构建与响应解析
"""

import json
import re

from common.config_loader import ThemeConfig, build_chunks_prompt


def build_extract_prompt(theme: ThemeConfig, chunks: list[dict]) -> str:
    """构造 extractor 的 prompt（注入维度列表）。"""
    template = build_chunks_prompt("step2_extract", theme, chunks)
    dims_desc = "\n".join(
        f"- `{d['key']}`: {d['desc']}"
        + (f"（可选值: {d['enum']}）" if "enum" in d else "")
        for d in theme.dimensions
    )
    return template.replace("{dimensions}", dims_desc)


_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.MULTILINE)


def parse_extract_response(raw: str, expected_ids: list[str]) -> dict[str, dict]:
    """解析 extractor 返回的 JSON 数组，按 id 对齐到 expected_ids。"""
    m = _JSON_BLOCK.search(raw)
    candidate = m.group(1).strip() if m else raw.strip()
    parsed = _try_parse(candidate)
    if parsed is None:
        start, end = candidate.find("["), candidate.rfind("]")
        if start >= 0 and end > start:
            parsed = _try_parse(candidate[start:end + 1])
    if parsed is None:
        return {}
    by_id = {str(r.get("id", "")).strip(): r for r in parsed}
    return {cid: by_id[cid] for cid in expected_ids if cid in by_id}


def _try_parse(text: str) -> list | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None