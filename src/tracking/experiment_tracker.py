from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.entities import Run


class ExperimentTracker:
    def __init__(self, experiment_name: str = "llm-eval-harness", tracking_uri: Optional[str] = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run: Optional[Run] = None

    def start_run(self, run_name: str, tags: Optional[dict[str, str]] = None) -> None:
        self._run = mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def log_dict(self, data: dict[str, Any], filename: str) -> None:
        mlflow.log_dict(data, filename)

    def set_tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)

    def end_run(self, status: str = "FINISHED") -> None:
        mlflow.end_run(status=status)

    def get_best_run(self, metric: str, mode: str = "max") -> dict[str, Any]:
        order = "DESC" if mode == "max" else "ASC"
        runs = mlflow.search_runs(
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
        if runs.empty:
            raise ValueError(f"No runs found with metric '{metric}'")
        return runs.iloc[0].to_dict()

    def get_run_metrics(self, run_name: str) -> dict[str, float]:
        runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'")
        if runs.empty:
            raise ValueError(f"No run found with name '{run_name}'")
        row = runs.iloc[0]
        return {
            col.replace("metrics.", ""): float(row[col])
            for col in runs.columns
            if col.startswith("metrics.") and row[col] is not None
        }

    @property
    def current_run_id(self) -> Optional[str]:
        run = mlflow.active_run()
        return run.info.run_id if run else None

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status=status)
