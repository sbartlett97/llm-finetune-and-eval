from __future__ import annotations

import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.schemas import RunConfig
from src.tracking.experiment_tracker import ExperimentTracker
from src.training.callbacks import TensorBoardStepCallback
from src.training.lora_config import build_lora_config
from src.types import TrainingResult

logger = logging.getLogger(__name__)


def _attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        logger.info("flash_attn not importable, falling back to sdpa")
        return "sdpa"


def _build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


class FineTuner:
    def __init__(
        self,
        config: RunConfig,
        train_dataset: Dataset,
        val_dataset: Dataset,
        tracker: ExperimentTracker,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tracker = tracker

    def train(self) -> TrainingResult:
        tc = self.config.training

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=_build_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=_attn_implementation(),
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.enable_input_require_grads()

        lora_config = build_lora_config(self.config.lora)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        output_dir = self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=tc.num_epochs,
            per_device_train_batch_size=tc.per_device_train_batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            lr_scheduler_type=tc.lr_scheduler_type,
            warmup_ratio=tc.warmup_ratio,
            max_grad_norm=tc.max_grad_norm,
            fp16=tc.fp16,
            bf16=tc.bf16,
            dataloader_num_workers=tc.dataloader_num_workers,
            seed=tc.seed,
            logging_steps=tc.logging_steps,
            eval_strategy="steps",
            eval_steps=tc.eval_steps,
            save_steps=tc.save_steps,
            save_total_limit=tc.save_total_limit,
            load_best_model_at_end=tc.load_best_model_at_end,
            metric_for_best_model=tc.metric_for_best_model,
            report_to="none",
            dataset_text_field="text",
            max_seq_length=self.config.training.max_seq_length,
            packing=True,
            loss_type="sft",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=tokenizer,
            callbacks=[TensorBoardStepCallback(self.tracker)],
        )

        train_result = trainer.train()
        trainer.save_model(output_dir)

        final_train_loss = train_result.training_loss
        eval_results = trainer.evaluate()
        final_val_loss = float(eval_results.get("eval_loss", 0.0))

        return TrainingResult(
            run_name=self.config.run_name,
            output_dir=output_dir,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            run_id=self.tracker.current_run_id or "",
        )

    def save_checkpoint(self, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
