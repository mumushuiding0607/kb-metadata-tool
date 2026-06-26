"""
model_local_qwen.py - 本地 Qwen 模型实现

支持 transformers 和 vllm 两种 backend，按 entry.extra.framework 选择。
"""

import time

from common.logger import get_logger
from common.model_interface import ModelEntry, ModelResponse

logger = get_logger("local_qwen")


class LocalQwenModel:
    """本地 Qwen 模型。"""

    def __init__(self, entry: ModelEntry):
        self.model_path = entry.extra.get("model_path", "")
        self.device = entry.extra.get("device", "cpu")
        self.framework = entry.extra.get("framework", "transformers")
        self.batch_size = entry.batch_size
        if not self.model_path:
            raise RuntimeError("local_qwen 模型需要 extra.model_path")
        self._model = None
        self._tokenizer = None
        self._torch = None  # 仅 transformers 后端需要
        self._load()

    def _load(self) -> None:
        if self.framework == "vllm":
            from vllm import LLM
            self._model = LLM(model=self.model_path, dtype="float16")
            logger.info("vllm 模型加载完成: %s", self.model_path)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=dtype
            ).to(self.device)
            logger.info("transformers 模型加载完成: %s (device=%s)",
                        self.model_path, self.device)

    def _infer(self, prompts: list[str]) -> list[str]:
        if self.framework == "vllm":
            outputs = self._model.generate(prompts, use_tqdm=False)
            return [o.outputs[0].text for o in outputs]
        assert self._torch is not None and self._tokenizer is not None
        results: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            inputs = self._tokenizer(batch, return_tensors="pt",
                                     padding=True).to(self.device)
            with self._torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=512)
            for j, prompt in enumerate(batch):
                gen = out[j][inputs["input_ids"].shape[1]:]
                results.append(self._tokenizer.decode(gen, skip_special_tokens=True))
        return results

    def generate(self, prompt: str, timeout: int = 30) -> ModelResponse:
        prompt_chars = len(prompt)
        logger.info("llm_call_start model=local_qwen prompt_chars=%d device=%s",
                    prompt_chars, self.device)
        start = time.monotonic()
        try:
            text = self._infer([prompt])[0]
        except Exception as e:
            logger.error("llm_call_error model=local_qwen err_type=%s err=%s",
                         type(e).__name__, e)
            raise
        latency = int((time.monotonic() - start) * 1000)
        logger.info("llm_call_ok model=local_qwen latency=%dms resp_chars=%d",
                    latency, len(text))
        return ModelResponse(text=text, latency_ms=latency)

    def batch_generate(self, prompts: list[str], timeout: int = 30) -> list[ModelResponse]:
        logger.info("llm_batch_start model=local_qwen n=%d batch_size=%d",
                    len(prompts), self.batch_size)
        start = time.monotonic()
        try:
            texts = self._infer(prompts)
        except Exception as e:
            logger.error("llm_batch_error model=local_qwen err_type=%s err=%s",
                         type(e).__name__, e)
            raise
        latency = int((time.monotonic() - start) * 1000)
        logger.info("llm_batch_done model=local_qwen n=%d latency=%dms",
                    len(prompts), latency)
        return [ModelResponse(text=t, latency_ms=latency) for t in texts]