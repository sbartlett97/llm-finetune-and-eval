#!/usr/bin/env python
"""Train a LoRA fine-tuned model.

Usage:
    python scripts/train.py --config configs/training/lora_r16.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    args = parser.parse_args()

    from src.config_loader import load_data_config, load_run_config
    from src.data.data_module import MedicalQADataModule
    from src.tracking.experiment_tracker import ExperimentTracker
    from src.training.fine_tuner import FineTuner
    from transformers import AutoTokenizer

    run_config = load_run_config(args.config)
    data_config = load_data_config(args.data_config)

    tracker = ExperimentTracker()
    tracker.start_run(
        run_name=run_config.run_name,
        tags={"model_type": "lora", "dataset_version": data_config.dataset_name},
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(run_config.base_model, use_fast=True)
        data_module = MedicalQADataModule(data_config, tokenizer=tokenizer)
        data_module.setup()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            stats = data_module.get_preprocessing_stats()
            json.dump(stats, f)
            tracker.log_artifact(f.name)

        tracker.log_params({
            "base_model": run_config.base_model,
            "lora_r": run_config.lora.r,
            "lora_alpha": run_config.lora.lora_alpha,
            "lora_dropout": run_config.lora.lora_dropout,
            "num_epochs": run_config.training.num_epochs,
            "learning_rate": run_config.training.learning_rate,
            "per_device_train_batch_size": run_config.training.per_device_train_batch_size,
            "gradient_accumulation_steps": run_config.training.gradient_accumulation_steps,
            "seed": run_config.training.seed,
            "train_size": data_config.train_size,
            "val_size": data_config.val_size,
            "dataset": data_config.dataset_name,
        })

        fine_tuner = FineTuner(
            config=run_config,
            train_dataset=data_module.get_train_dataset(),
            val_dataset=data_module.get_val_dataset(),
            tracker=tracker,
        )
        result = fine_tuner.train()

        tracker.log_metrics({
            "final_train_loss": result.final_train_loss,
            "final_val_loss": result.final_val_loss,
        })
        tracker.set_tag("model_type", "lora")
        logger.info("Training complete: %s", result)
        tracker.end_run("FINISHED")

    except Exception:
        logger.exception("Training failed")
        tracker.end_run("FAILED")
        raise


if __name__ == "__main__":
    main()
