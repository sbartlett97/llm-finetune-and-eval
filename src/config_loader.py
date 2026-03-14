from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from src.schemas import DataConfig, EvalConfig, RunConfig

T = TypeVar("T", bound=BaseModel)


def _load_yaml(path: str | Path) -> dict:  # type: ignore[type-arg]
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path: str | Path, model: type[T]) -> T:
    return model.model_validate(_load_yaml(path))


def load_run_config(path: str | Path) -> RunConfig:
    return load_config(path, RunConfig)


def load_data_config(path: str | Path) -> DataConfig:
    return load_config(path, DataConfig)


def load_eval_config(path: str | Path) -> EvalConfig:
    return load_config(path, EvalConfig)
