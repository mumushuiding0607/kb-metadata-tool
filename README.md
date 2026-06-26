# kb-metadata-tool

知识库文本块 → 元数据 + HyDE 提取流水线。

支持**主题可插拔**（切换 AI变现 / 炒股）、**模型可插拔**（切换本地/云端）、**断点续传**、**限流友好**。

## 架构

```
原始分块 → [filter 本地粗筛] → 02_filtered.jsonl
                                  ↓
                       [extractor 云端精炼] → 03_extracted.jsonl
                                                ↓
                                       [hyde 高质量块] → 04_hyde.jsonl
                                                              ↓
                                                  [merge 合并] → 05_final_output.json
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 MINIMAX_API_KEY

# 3. 放置输入数据
cp your_chunks.json data/input/01_raw_chunks.json

# 4. 执行
python run.py                        # 跑完全流程
python run.py --step filter          # 只跑第一步
python run.py --theme config/theme/stock.json  # 切换主题
python run.py --model config/models/all_gpt4.yaml  # 切换模型
```

## 断点续传

每步都自动跳过已完成块。中断后直接重新执行 `python run.py` 即可。

失败批次写入 `data/checkpoints/0*_pending.jsonl`，下次自动重试。

## 项目结构

- `common/` - 基础设施（日志、配置、模型接口、限流、文件 IO）
- `modules/` - 业务模块（filter / extractor / hyde_generator / merge_outputs）
- `config/` - 主题、模型、Prompt 模板
- `data/` - 输入、checkpoint、输出
- `run.py` - 主编排器