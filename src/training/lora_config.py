from __future__ import annotations

from peft import LoraConfig as PeftLoraConfig, TaskType

from src.schemas import LoraConfig


def build_lora_config(config: LoraConfig) -> PeftLoraConfig:
    return PeftLoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        task_type=TaskType.CAUSAL_LM,
    )
