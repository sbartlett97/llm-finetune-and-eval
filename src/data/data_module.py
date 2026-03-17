from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from src.data.preprocessing import (
    compute_preprocessing_stats,
    format_prompt,
    passes_length_filter,
    passes_quality_filter,
)
from src.schemas import DataConfig

logger = logging.getLogger(__name__)

SPLIT_INDICES_FILENAME = "split_indices.json"


class MedicalQADataModule:
    def __init__(self, config: DataConfig, tokenizer: Optional[Any] = None):
        self.config = config
        self.tokenizer = tokenizer
        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None
        self._preprocessing_stats: dict[str, int] = {}

    def setup(self, split_indices_path: Optional[str] = None) -> None:
        raw = load_dataset(self.config.dataset_name, split="train")
        assert isinstance(raw, Dataset)

        initial_count = len(raw)

        raw = raw.filter(passes_quality_filter)
        after_quality = len(raw)

        if self.tokenizer is not None:
            raw = raw.filter(
                lambda r: passes_length_filter(
                    r,
                    self.tokenizer,
                    self.config.min_input_tokens,
                    self.config.min_output_tokens,
                    self.config.max_length,
                )
            )
        after_length = len(raw)

        self._preprocessing_stats = compute_preprocessing_stats(
            initial_count, after_quality, after_length
        )
        logger.info("Preprocessing stats: %s", self._preprocessing_stats)

        total = self.config.train_size + self.config.val_size + self.config.test_size
        assert len(raw) >= total, f"Dataset too small after filtering: {len(raw)} < {total}"

        raw = raw.map(format_prompt)

        if split_indices_path and Path(split_indices_path).exists():
            self._load_splits_from_indices(raw, split_indices_path)
        else:
            self._create_splits(raw, split_indices_path)

    def _create_splits(self, dataset: Dataset, save_path: Optional[str]) -> None:
        test_val_size = self.config.val_size + self.config.test_size
        split1 = dataset.train_test_split(
            test_size=test_val_size, seed=self.config.seed, shuffle=True
        )
        split2 = split1["test"].train_test_split(
            test_size=self.config.test_size, seed=self.config.seed
        )

        self._train = split1["train"].select(range(self.config.train_size))
        self._val = split2["train"]
        self._test = split2["test"]

        if save_path:
            indices = {
                "train": self._train["__index_level_0__"]
                if "__index_level_0__" in self._train.column_names
                else list(range(len(self._train))),
            }
            Path(save_path).write_text(json.dumps(indices))

    def _load_splits_from_indices(self, dataset: Dataset, path: str) -> None:
        indices = json.loads(Path(path).read_text())
        test_val_size = self.config.val_size + self.config.test_size
        split1 = dataset.train_test_split(
            test_size=test_val_size, seed=self.config.seed, shuffle=True
        )
        split2 = split1["test"].train_test_split(
            test_size=self.config.test_size, seed=self.config.seed
        )
        self._train = split1["train"].select(range(self.config.train_size))
        self._val = split2["train"]
        self._test = split2["test"]

    def _check_setup(self) -> None:
        if self._train is None:
            raise RuntimeError("Call setup() first")

    def get_train_dataset(self) -> Dataset:
        self._check_setup()
        assert self._train is not None
        return self._train

    def get_val_dataset(self) -> Dataset:
        self._check_setup()
        assert self._val is not None
        return self._val

    def get_test_dataset(self) -> Dataset:
        self._check_setup()
        assert self._test is not None
        return self._test

    def get_preprocessing_stats(self) -> dict[str, int]:
        return self._preprocessing_stats
