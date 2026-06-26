"""
run.py - 主编排器

CLI:
  python run.py [--step STEP] [--theme PATH] [--model PATH] [--input PATH]
"""

import argparse
import sys
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config_loader import load_models, load_theme
from common.logger import get_logger
from modules import extractor, filter, hyde_generator, merge_outputs

logger = get_logger("run")


class Step(str, Enum):
    FILTER = "filter"
    EXTRACT = "extract"
    HYDE = "hyde"
    MERGE = "merge"
    ALL = "all"


_PIPELINE: dict[Step, list] = {
    Step.FILTER: [filter.run],
    Step.EXTRACT: [extractor.run],
    Step.HYDE: [hyde_generator.run],
    Step.MERGE: [merge_outputs.run],
    Step.ALL: [filter.run, extractor.run, hyde_generator.run, merge_outputs.run],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="kb-metadata-tool 主编排器")
    parser.add_argument("--step", default=Step.ALL, type=Step, choices=list(Step))
    parser.add_argument("--theme", default=None, help="主题配置 JSON 路径")
    parser.add_argument("--model", default=None, help="模型配置 YAML 路径")
    parser.add_argument("--input", default=None, help="输入数据文件路径")
    args = parser.parse_args()

    theme = load_theme(args.theme)
    models = load_models(args.model)
    logger.info("===== 启动 step=%s, theme=%s =====", args.step.value, theme.name)

    for fn in _PIPELINE[args.step]:
        if fn is filter.run:
            fn(theme=theme, models=models, input_path=args.input)
        else:
            fn(theme=theme, models=models)

    logger.info("===== 全部完成 =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())