from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self) -> None:
        self.model: object = None
        self.tokenizer: object = None
        self.model_run_id: str = "unknown"
        self._ready = False

    def load(self) -> None:
        from unsloth import FastLanguageModel

        model_path = self._resolve_model_path()
        logger.info("Loading model from: %s", model_path)

        max_seq_length = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
        # Unsloth auto-detects LoRA adapters (adapter_config.json) and loads
        # the correct base model before applying the adapter.
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore[union-attr]
        FastLanguageModel.for_inference(self.model)  # 2x faster inference kernels
        self._ready = True
        logger.info("Model ready")

    def _resolve_model_path(self) -> str:
        if path := os.environ.get("MODEL_PATH"):
            return path
        if run_id := os.environ.get("MODEL_RUN_ID"):
            self.model_run_id = run_id
            checkpoint_dir = Path("checkpoints") / run_id
            if not checkpoint_dir.exists():
                raise RuntimeError(
                    f"Checkpoint directory '{checkpoint_dir}' not found. "
                    "Set MODEL_PATH to an explicit path or ensure the checkpoint exists."
                )
            return str(checkpoint_dir)
        raise RuntimeError("Set MODEL_PATH or MODEL_RUN_ID environment variable")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def generate(
        self,
        question: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> tuple[str, int]:
        from src.data.preprocessing import format_prompt_inference

        prompt = format_prompt_inference(question, self.tokenizer)

        inputs = self.tokenizer(  # type: ignore[call-arg]
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)  # type: ignore[union-attr]

        with torch.no_grad():
            output = self.model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,  # type: ignore[union-attr]
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)  # type: ignore[union-attr]
        return text.strip(), len(generated_ids)
