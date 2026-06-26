"""
batch_runner.py - 通用批处理编排器

业务模块只调用 run_pipeline()，不重复实现 pending/batch loop 骨架。
"""

from itertools import chain
from pathlib import Path
from typing import Callable

from common import file_utils
from common.logger import get_logger

logger = get_logger("batch_runner")


def run_pipeline(
    chunks: list[dict],
    *,
    process_batch: Callable[[list[dict]], dict[str, dict]],
    success_path: Path,
    pending_path: Path,
    build_success: Callable[[dict, dict], dict],
    build_pending: Callable[[dict, str], dict],
    batch_size: int,
    description: str = "",
) -> tuple[int, int]:
    """通用批处理编排。

    Args:
        chunks: 待处理块列表
        process_batch: 输入一批，返回 {id: result_dict}；失败抛异常
        success_path: 成功结果写入此 JSONL
        pending_path: 失败批次写入此 JSONL
        build_success: (chunk, result_dict) → success record（返回 None 表示跳过）
        build_pending: (chunk, reason) → pending record
        batch_size: 每批大小
        description: 用于日志描述

    Returns:
        (success_count, fail_count)
    """
    if not chunks:
        logger.info("%s 无待处理块，退出", description or "pipeline")
        return 0, 0

    completed: set[str] = set()
    for r in chain(file_utils.read_jsonl(success_path),
                   file_utils.read_jsonl(pending_path)):
        if "id" in r:
            completed.add(r["id"])
    pending = [c for c in chunks if c["id"] not in completed]
    logger.info("%s 已完成 %d，跳过；待处理 %d",
                description, len(completed), len(pending))
    if not pending:
        return 0, 0

    success_count = fail_count = 0
    skipped_count = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        ids = [c["id"] for c in batch]
        logger.debug("batch_call desc=%s idx=%d size=%d ids=%s",
                     description, i, len(batch), ids[:3])
        try:
            results = process_batch(batch)
        except Exception as e:
            logger.error("%s batch_failed ids=%s err_type=%s err=%s",
                         description, ids, type(e).__name__, e)
            for c in batch:
                file_utils.append_jsonl(pending_path, build_pending(c, str(e)))
            fail_count += len(batch)
            continue

        for c in batch:
            r = results.get(c["id"])
            if not r:
                file_utils.append_jsonl(pending_path, build_pending(c, "missing"))
                fail_count += 1
                continue
            # build_success 返回 None 表示跳过（例如 filter 丢弃 unrelated）
            success_record = build_success(c, r)
            if success_record is None:
                skipped_count += 1
                continue
            file_utils.append_jsonl(success_path, success_record)
            success_count += 1
        logger.info("%s 进度 %d/%d", description,
                    min(i + batch_size, len(pending)), len(pending))

    logger.info("%s 完成: 成功 %d，跳过 %d，失败 %d",
                description, success_count, skipped_count, fail_count)
    return success_count, fail_count
