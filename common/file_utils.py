"""
file_utils.py - JSONL 文件流式读写与断点恢复

设计原则：
- 全部用 JSONL（一行一条 JSON），避免大数组 JSON 占用内存。
- 追加写（mode='a'）保证已完成批次不会被破坏。
- read_completed_ids() 自动跳过空行和损坏行。
"""

import json
from pathlib import Path
from typing import Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """流式读取 JSONL 文件，跳过空行/损坏行。文件不存在返回空迭代器。"""
    p = Path(path)
    try:
        f = p.open("r", encoding="utf-8")
    except FileNotFoundError:
        return
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 单行损坏不应阻断整体读取
                continue


def append_jsonl(path: str | Path, record: dict) -> None:
    """追加一条 JSON 记录到 JSONL 文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """覆盖写入整个 JSONL 文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_completed_ids(path: str | Path, id_field: str = "id") -> set[str]:
    """从 checkpoint 文件读取所有已完成 ID。"""
    return {r[id_field] for r in read_jsonl(path) if id_field in r}


def read_json(path: str | Path) -> dict | list | None:
    """读取单对象 JSON 文件，不存在返回 None。"""
    p = Path(path)
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def write_json(path: str | Path, data) -> None:
    """覆盖写入 JSON 文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)