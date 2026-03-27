#!/usr/bin/env python
"""Push trained LoRA checkpoints to Hugging Face Hub.

Usage:
    # Push a single run (LoRA adapter only):
    python scripts/push_to_hub.py --run-id run_lora_r16 --hf-repo your-username/qwen2.5-3b-medical

    # Push merged (full 16-bit) model:
    python scripts/push_to_hub.py --run-id run_lora_r16 --hf-repo your-username/qwen2.5-3b-medical --merge

    # Push all experiment runs:
    python scripts/push_to_hub.py --all --hf-repo-prefix your-username/qwen2.5-3b-medical
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

EXPERIMENT_RUNS = [
    "run_lora_r8",
    "run_lora_r16",
    "run_lora_r32",
    "run_lora_r16_2ep",
]


def push_run(
    run_id: str,
    hf_repo: str,
    merge: bool,
    hf_token: str | None,
    private: bool,
) -> None:
    checkpoint_dir = Path("checkpoints") / run_id
    if not checkpoint_dir.exists():
        logger.warning("Checkpoint not found, skipping: %s", checkpoint_dir)
        return

    logger.info("Loading checkpoint: %s", checkpoint_dir)

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_dir),
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )

    push_kwargs: dict[str, object] = {
        "private": private,
        **({"token": hf_token} if hf_token else {}),
    }

    if merge:
        logger.info("Merging adapter and pushing full 16-bit model to %s", hf_repo)
        model.save_pretrained_merged(
            hf_repo,
            tokenizer,
            save_method="merged_16bit",
            push_to_hub=True,
            **push_kwargs,
        )
    else:
        logger.info("Pushing LoRA adapter to %s", hf_repo)
        model.push_to_hub(hf_repo, **push_kwargs)
        tokenizer.push_to_hub(hf_repo, **push_kwargs)

    logger.info("Done: %s → %s", run_id, hf_repo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Push checkpoints to Hugging Face Hub")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Single run ID (e.g. run_lora_r16)")
    group.add_argument("--all", action="store_true", help="Push all experiment runs")

    parser.add_argument(
        "--hf-repo",
        help="Target Hub repo (required with --run-id). E.g. username/model-name",
    )
    parser.add_argument(
        "--hf-repo-prefix",
        help="Repo prefix for --all mode. Each run appended as suffix. E.g. username/qwen2.5-3b-medical",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA adapter into base weights and push full 16-bit model",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create private Hub repo (default: public)",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set — Hub push will use cached credentials only")

    if args.run_id:
        if not args.hf_repo:
            parser.error("--hf-repo is required with --run-id")
        push_run(args.run_id, args.hf_repo, args.merge, hf_token, args.private)

    else:  # --all
        if not args.hf_repo_prefix:
            parser.error("--hf-repo-prefix is required with --all")
        for run_id in EXPERIMENT_RUNS:
            repo = f"{args.hf_repo_prefix}-{run_id}"
            push_run(run_id, repo, args.merge, hf_token, args.private)


if __name__ == "__main__":
    main()
