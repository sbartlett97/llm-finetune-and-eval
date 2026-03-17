from __future__ import annotations

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.tracking.experiment_tracker import ExperimentTracker


class TensorBoardStepCallback(TrainerCallback):
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        if logs is None:
            return
        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if metrics:
            self.tracker.log_metrics(metrics, step=state.global_step)
