from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

_ADAPTER_CONFIG = "adapter_config.json"


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


class ModelLoader:
    def __init__(self) -> None:
        self.model: object = None
        self.tokenizer: object = None
        self.model_run_id: str = "unknown"
        self._ready = False

    def load(self) -> None:
        model_path = self._resolve_model_path()
        logger.info("Loading model from: %s", model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore[union-attr]

        if (Path(model_path) / _ADAPTER_CONFIG).exists():
            self._load_lora(model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=_bnb_config(),
                device_map="auto",
            )

        self.model.eval()  # type: ignore[union-attr]
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

    def _load_lora(self, adapter_path: str) -> None:
        from peft import PeftModel
        config_path = Path(adapter_path) / _ADAPTER_CONFIG
        base_model_id = json.loads(config_path.read_text())["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=_bnb_config(),
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)

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
        from src.data.preprocessing import PROMPT_TEMPLATE

        prompt = PROMPT_TEMPLATE.format(input=question, output="")
        prompt = prompt.rsplit("[/INST]", 1)[0] + "[/INST]"

        inputs = self.tokenizer(  # type: ignore[call-arg]
            prompt, return_tensors="pt", truncation=True, max_length=512
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
