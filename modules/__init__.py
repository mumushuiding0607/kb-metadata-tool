"""
modules - 业务模块

业务模块只组合调用 common/ 提供的基础设施，不自行实现。
- filter: 第一步（本地模型粗筛）
- extractor: 第二步（高级模型精炼）
- hyde_generator: 第三步（HyDE 生成）
- merge_outputs: 最终合并
"""