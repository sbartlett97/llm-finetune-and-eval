#!/usr/bin/env python
"""Run the full eval suite against a model checkpoint.

Usage:
    python scripts/eval.py --run-id run_lora_r16
    python scripts/eval.py --model-path ./checkpoints/run_lora_r16
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Run name matching a directory under runs/ (e.g. run_lora_r16)")
    group.add_argument("--model-path", help="Local path to model checkpoint")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    args = parser.parse_args()

    from src.config_loader import load_data_config, load_eval_config
    from src.data.data_module import MedicalQADataModule
    from src.evaluation.eval_runner import EvalRunner
    from src.tracking.experiment_tracker import ExperimentTracker

    data_config = load_data_config(args.data_config)
    eval_config = load_eval_config(args.eval_config)

    data_module = MedicalQADataModule(data_config)
    data_module.setup()
    test_dataset = data_module.get_test_dataset()

    run_id = args.run_id or Path(args.model_path).name
    model_path = args.model_path or f"./checkpoints/{args.run_id}"

    tracker = ExperimentTracker()
    tracker.start_run(
        run_name=f"eval_{run_id}",
        tags={"model_type": "eval", "evaluated_run": run_id},
    )

    try:
        runner = EvalRunner(
            model_path=model_path,
            config=eval_config,
            tracker=tracker,
            run_id=run_id,
        )
        report = runner.run(test_dataset)

        output_dir = Path("reports") / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        import dataclasses, json
        (output_dir / "eval_report.json").write_text(
            json.dumps(dataclasses.asdict(report), indent=2)
        )
        logger.info("Eval complete. Report saved to %s", output_dir)

        tracker.log_artifact(str(output_dir / "eval_report.json"))
        tracker.end_run("FINISHED")

    except Exception:
        logger.exception("Eval failed")
        tracker.end_run("FAILED")
        raise


if __name__ == "__main__":
    main()
