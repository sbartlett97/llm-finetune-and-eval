from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from src.schemas import RunConfig
from src.tracking.experiment_tracker import ExperimentTracker
from src.training.callbacks import TensorBoardStepCallback
from src.types import TrainingResult

logger = logging.getLogger(__name__)


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
        from unsloth import FastLanguageModel

        tc = self.config.training
        lc = self.config.lora

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=tc.max_seq_length,
            load_in_4bit=True,
            dtype=None,  # auto: bf16 on Ampere+, fp16 on older hardware
        )
        # Qwen2.5 uses ChatML natively — <|im_end|> is already the EOS token in
        # its Rust vocab, so no template remapping or token registration is needed.
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = FastLanguageModel.get_peft_model(
            model,
            r=lc.r,
            lora_alpha=lc.lora_alpha,
            lora_dropout=lc.lora_dropout,
            target_modules=lc.target_modules,
            bias=lc.bias,
            use_gradient_checkpointing="unsloth",  # ~30% less VRAM; supports long contexts
            random_state=tc.seed,
        )
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
            packing=True,
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

