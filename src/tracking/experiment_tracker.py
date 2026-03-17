from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from torch.utils.tensorboard import SummaryWriter


class ExperimentTracker:
    """TensorBoard-backed experiment tracker.

    Each run is written to ``{log_dir}/{run_name}/``.  Scalar metrics go into
    TensorBoard event files; params, tags, and JSON artefacts are stored as
    sidecar files in the same directory so the whole run folder is
    self-contained.

    ``get_run_metrics`` and ``get_best_run`` read from ``metrics_summary.json``
    written at ``end_run``, enabling cross-run comparison without a running
    tracking server.

    Visualise with::

        tensorboard --logdir runs/
    """

    def __init__(self, log_dir: str = "runs"):
        self._log_dir = Path(log_dir)
        self._run_name: Optional[str] = None
        self._writer: Optional[SummaryWriter] = None
        self._params: dict[str, Any] = {}
        self._tags: dict[str, str] = {}
        self._metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_name: str, tags: Optional[dict[str, str]] = None) -> None:
        self._run_name = run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self._run_dir))
        self._params = {}
        self._tags = dict(tags or {})
        self._metrics = {}

    def end_run(self, status: str = "FINISHED") -> None:
        if self._writer is None:
            return
        self._tags["status"] = status
        self._run_dir.joinpath("params.json").write_text(
            json.dumps(self._params, indent=2)
        )
        self._run_dir.joinpath("tags.json").write_text(
            json.dumps(self._tags, indent=2)
        )
        self._run_dir.joinpath("metrics_summary.json").write_text(
            json.dumps(self._metrics, indent=2)
        )
        self._writer.flush()
        self._writer.close()
        self._writer = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        self._params.update(params)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        assert self._writer is not None, "call start_run() first"
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, global_step=step)
            # Keep the last value so metrics_summary.json is always up to date.
            self._metrics[name] = value

    def log_artifact(self, path: str | Path) -> None:
        src = Path(path)
        dest = self._run_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)

    def log_dict(self, data: dict[str, Any], filename: str) -> None:
        self._run_dir.joinpath(filename).write_text(json.dumps(data, indent=2))

    def set_tag(self, key: str, value: str) -> None:
        self._tags[key] = value

    # ------------------------------------------------------------------
    # Cross-run queries (read metrics_summary.json written by end_run)
    # ------------------------------------------------------------------

    def get_run_metrics(self, run_name: str) -> dict[str, float]:
        summary = self._log_dir / run_name / "metrics_summary.json"
        if not summary.exists():
            raise ValueError(f"No completed run found at '{summary}'")
        return json.loads(summary.read_text())

    def get_best_run(self, metric: str, mode: str = "max") -> dict[str, Any]:
        best_value: Optional[float] = None
        best_run: Optional[dict[str, Any]] = None
        for summary_path in self._log_dir.glob("*/metrics_summary.json"):
            metrics = json.loads(summary_path.read_text())
            if metric not in metrics:
                continue
            value = metrics[metric]
            if best_value is None or (mode == "max" and value > best_value) or (mode == "min" and value < best_value):
                best_value = value
                best_run = {"run_name": summary_path.parent.name, **metrics}
        if best_run is None:
            raise ValueError(f"No runs found with metric '{metric}'")
        return best_run

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def current_run_id(self) -> Optional[str]:
        return self._run_name

    @property
    def _run_dir(self) -> Path:
        assert self._run_name is not None, "call start_run() first"
        return self._log_dir / self._run_name

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_run("FAILED" if exc_type else "FINISHED")
