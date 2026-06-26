"""
run.py - 主编排器

每个 step 都可以独立执行，以上一步的产物为输入。
所有产物集中存放在 data/<input_filename>/ 目录下。

CLI:
  python run.py [--step STEP] [--input PATH] [--theme PATH] [--model PATH]

示例:
  # 全流程
  python run.py --input data/input/01_raw_chunks.json

  # 单步执行（每次都传原始输入，自动定位到 run_dir）
  python run.py --step filter  --input data/input/01_raw_chunks.json
  python run.py --step extract --input data/input/01_raw_chunks.json
  python run.py --step hyde    --input data/input/01_raw_chunks.json
  python run.py --step merge   --input data/input/01_raw_chunks.json

  # 也可直接指向 run_dir 下的产物
  python run.py --step extract --input data/01_raw_chunks/02_filtered.jsonl
"""

import argparse
import sys
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config_loader import (
    DATA_DIR, RunPaths, derive_run_dir, load_models, load_theme,
)
from common.logger import get_logger
from modules import extractor, filter, hyde_generator, merge_outputs

logger = get_logger("run")


class Step(str, Enum):
    FILTER = "filter"
    EXTRACT = "extract"
    HYDE = "hyde"
    MERGE = "merge"
    ALL = "all"


# step → (fn, 是否需要 input_path)。filter 之后的步骤可独立跑。
_PIPELINE: dict[Step, tuple] = {
    Step.FILTER: (filter.run, True),
    Step.EXTRACT: (extractor.run, True),
    Step.HYDE: (hyde_generator.run, True),
    Step.MERGE: (merge_outputs.run, True),
    Step.ALL: (None, True),  # all 模式：按顺序跑全部
}


def _run_all(run_dir: Path, theme, models) -> None:
    for step in (Step.FILTER, Step.EXTRACT, Step.HYDE, Step.MERGE):
        _PIPELINE[step][0](theme=theme, models=models, run_dir=run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="kb-metadata-tool 主编排器")
    parser.add_argument("--step", default=Step.ALL, type=Step, choices=list(Step))
    parser.add_argument("--theme", default=None, help="主题配置 JSON 路径")
    parser.add_argument("--model", default=None, help="模型配置 YAML 路径")
    parser.add_argument("--input", default=None,
                        help="原始分块文件路径（或 run_dir 下的产物路径）")
    args = parser.parse_args()

    theme = load_theme(args.theme)
    models = load_models(args.model)

    if args.input is None:
        raise SystemExit("必须提供 --input 参数（原始 chunk 文件或 run_dir 产物路径）")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    run_dir = derive_run_dir(input_path)
    RunPaths.for_run_dir(run_dir)  # 确保目录存在

    logger.info("===== 启动 step=%s, run_dir=%s =====", args.step.value, run_dir)

    fn, needs_input = _PIPELINE[args.step]
    if fn is None:
        _run_all(run_dir, theme, models)
    elif needs_input:
        fn(theme=theme, models=models, input_path=input_path)
    else:
        fn(theme=theme, models=models, run_dir=run_dir)

    logger.info("===== 全部完成 =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())