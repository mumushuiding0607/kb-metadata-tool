"""
run_context.py - 各 step 入口的通用初始化

每个 step 函数前 5-7 行是相同的：加载 theme/models、推导 run_dir、构造
RunPaths。抽到这里避免在 filter/extractor/hyde/merge 四处重复。
"""

from dataclasses import dataclass
from pathlib import Path

from common.config_loader import (
    ModelConfig, RunPaths, ThemeConfig,
    derive_run_dir, load_models, load_theme,
)


@dataclass
class RunContext:
    theme: ThemeConfig
    models: ModelConfig
    paths: RunPaths


def setup_run(
    step_name: str,
    *,
    theme: ThemeConfig | None = None,
    models: ModelConfig | None = None,
    input_path: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> RunContext:
    """加载 theme/models、推导 run_dir、构造 RunPaths。

    Args:
        step_name: 用于错误信息
        theme: 预加载主题；为 None 时从配置读取
        models: 预加载模型；为 None 时从配置读取
        input_path: 原始输入或 run_dir 产物路径（与 run_dir 二选一）
        run_dir: 已知的 run_dir 路径（优先于 input_path）

    Returns:
        RunContext（theme, models, paths）
    """
    resolved_theme = theme or load_theme()
    resolved_models = models or load_models()
    if run_dir is None:
        if input_path is None:
            raise ValueError(f"{step_name} 需要 input_path 或 run_dir")
        run_dir = derive_run_dir(input_path)
    return RunContext(
        theme=resolved_theme,
        models=resolved_models,
        paths=RunPaths.for_run_dir(run_dir),
    )
