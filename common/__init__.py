"""
common - 基础设施层

业务模块（modules/）禁止实现以下能力，统一从这里获取：
- logger: 日志
- file_utils: JSONL 读写
- config_loader: 配置加载
- model_interface + model_factory: 模型调用
- token_bucket: 限流
"""