from __future__ import annotations

import logging
from threading import Lock

from backend.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._model = None
        self._tokenizer = None
        self._transformers = None
        self._torch = None
        self._loaded = False
        self._fallback_mode = False
        self._model_name = settings.model_id

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                quantization = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )

                logger.info("Loading model %s with 4-bit quantization", settings.model_id)
                tokenizer = AutoTokenizer.from_pretrained(settings.model_id, token=settings.hf_token)
                model = AutoModelForCausalLM.from_pretrained(
                    settings.model_id,
                    token=settings.hf_token,
                    quantization_config=quantization,
                    device_map="auto",
                )
                self._model = model
                self._tokenizer = tokenizer
                self._transformers = __import__("transformers")
                self._torch = torch
                self._fallback_mode = False
                logger.info("Model loaded successfully in quantized mode")
            except Exception as exc:
                logger.warning("Falling back from quantized load: %s", exc)
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(settings.model_id, token=settings.hf_token)
                    model = AutoModelForCausalLM.from_pretrained(
                        settings.model_id,
                        token=settings.hf_token,
                        device_map="cpu",
                    )
                    self._model = model
                    self._tokenizer = tokenizer
                    self._transformers = __import__("transformers")
                    self._torch = torch
                    self._fallback_mode = False
                    logger.info("Model loaded on CPU fallback")
                except Exception as fallback_exc:
                    self._fallback_mode = True
                    logger.error("Model unavailable, using deterministic fallback: %s", fallback_exc)
            finally:
                self._loaded = True

    @property
    def fallback_mode(self) -> bool:
        self._load()
        return self._fallback_mode

    @property
    def model_name(self) -> str:
        return self._model_name

    def _build_prompt(self, task: str, user_input: str, context: list[str] | None = None) -> str:
        context_block = "\n\n".join(context or [])
        if context_block:
            return f"Task: {task}\n\nContext:\n{context_block}\n\nUser Input:\n{user_input}\n\nResponse:"
        return f"Task: {task}\n\nUser Input:\n{user_input}\n\nResponse:"

    def _fallback_generate(self, prompt: str, summarize: bool = False) -> str:
        if summarize:
            cleaned = " ".join(prompt.split())
            return cleaned[:400] + ("..." if len(cleaned) > 400 else "")
        return "I could not load the configured model, so this is fallback mode. Prompt received: " + prompt[:300]

    def generate(self, *, task: str, user_input: str, context: list[str] | None = None, max_new_tokens: int = 256) -> str:
        self._load()
        prompt = self._build_prompt(task=task, user_input=user_input, context=context)
        if self._fallback_mode:
            return self._fallback_generate(prompt, summarize=(task == "summarize"))

        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        encoded = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(self._model, "device"):
            encoded = {k: v.to(self._model.device) for k, v in encoded.items()}

        with self._torch.no_grad():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        decoded = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()
        return response or "No response generated."


llm_service = LLMService()
