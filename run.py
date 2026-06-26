"""
run.py - 主编排器

CLI:
  python run.py [--step STEP] [--theme PATH] [--model PATH] [--input PATH]

STEP:
  all      - 跑完全部三步 + 合并（默认）
  filter   - 只跑第一步
  extract  - 只跑第二步（依赖 filter 输出）
  hyde     - 只跑第三步（依赖 extract 输出）
  merge    - 只合并 checkpoint
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config_loader import load_models, load_theme
from common.logger import get_logger
from modules import extractor, filter, hyde_generator, merge_outputs

logger = get_logger("run")


_STEP_DISPATCH = {
    "filter": [filter.run],
    "extract": [extractor.run],
    "hyde": [hyde_generator.run],
    "merge": [merge_outputs.run],
    "all": [filter.run, extractor.run, hyde_generator.run, merge_outputs.run],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="kb-metadata-tool 主编排器")
    parser.add_argument("--step", default="all", choices=list(_STEP_DISPATCH))
    parser.add_argument("--theme", default=None, help="主题配置 JSON 路径")
    parser.add_argument("--model", default=None, help="模型配置 YAML 路径")
    parser.add_argument("--input", default=None, help="输入数据文件路径")
    args = parser.parse_args()

    theme = load_theme(args.theme)
    models = load_models(args.model)
    logger.info("===== 启动 step=%s, theme=%s =====", args.step, theme.name)

    for fn in _STEP_DISPATCH[args.step]:
        if fn is filter.run:
            fn(theme=theme, models=models, input_path=args.input)
        else:
            fn(theme=theme, models=models)

    logger.info("===== 全部完成 =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())