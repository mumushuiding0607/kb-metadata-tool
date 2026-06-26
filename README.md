# kb-metadata-tool

知识库文本块 → 元数据 + HyDE 提取流水线。

支持**主题可插拔**、**模型可插拔**、**断点续传**、**限流友好**。

## 架构

每个输入文件对应一个 `run_dir`（默认 `data/<input_stem>/`），所有产物集中存放：

```
data/input/01_raw_chunks.json             ← 原始分块输入
        ↓
data/01_raw_chunks/
   ├── 02_filtered.jsonl                  ← filter 输出（已剔除 unrelated）
   ├── 02_pending.jsonl                   ← filter 失败批次（重试用）
   ├── 03_extracted.jsonl                 ← extractor 成功
   ├── 03_pending.jsonl                   ← extractor 失败
   ├── 04_hyde.jsonl                      ← hyde 成功
   ├── 04_pending.jsonl                   ← hyde 失败
   ├── 05_final_output.json               ← merge 全量
   └── 05_final_output.jsonl              ← merge 流式
```

## 快速开始

```bash
# 1. 安装依赖 + 配置 API Key
pip install -r requirements.txt
cp .env.example .env   # 编辑填入 MINIMAX_API_KEY

# 2. 准备输入
mkdir -p data/my_articles
cp your_chunks.json data/my_articles/01_raw_chunks.json

# 3. 全流程
python run.py --input data/my_articles/01_raw_chunks.json
```

## 单步执行

每一步都可以独立运行，跑过的会自动跳过（断点续传）：

```bash
python run.py --step filter  --input data/my_articles/01_raw_chunks.json
python run.py --step extract --input data/my_articles/01_raw_chunks.json
python run.py --step hyde    --input data/my_articles/01_raw_chunks.json
python run.py --step merge   --input data/my_articles/01_raw_chunks.json
```

也可以直接指向 `run_dir` 下的产物路径（自动识别）：

```bash
python run.py --step extract --input data/my_articles/02_filtered.jsonl
```

## 切换主题/模型

```bash
python run.py --input data/my_articles/01_raw_chunks.json \
              --theme config/theme/stock.json \
              --model config/models/all_gpt4.yaml
```

## 业务流程

1. **filter**：本地 Qwen 粗筛，每个块打 `relevance`（direct/inspirational/unrelated）+ `rough_density`（high/medium/low）。**unrelated 直接丢弃**，不再进入下游。
2. **extractor**：对 medium/high 密度的相关块，调云端模型提取主题维度的元数据 + density_score。
3. **hyde_generator**：对 density_score ≥ threshold 的块生成 ≤50 字的假设性问题。
4. **merge_outputs**：合并所有 checkpoint 输出最终 JSON/JSONL。

## 项目结构

- `common/` - 基础设施（日志、配置、模型接口、限流、文件 IO、重试、批处理编排）
- `modules/` - 业务模块（filter / extractor / hyde_generator / merge_outputs）
- `config/` - 主题、模型、Prompt 模板
- `run.py` - 主编排器 + CLI
- `data/` - 输入、checkpoint（按 run_dir 分散）、输出