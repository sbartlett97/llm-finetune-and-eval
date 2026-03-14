from __future__ import annotations

from src.schemas import RegressionThresholds
from src.types import AutomatedMetricResults, JudgeResults


def detect_regressions(
    current: AutomatedMetricResults,
    baseline: AutomatedMetricResults,
    thresholds: RegressionThresholds,
    current_judge: JudgeResults | None = None,
    baseline_judge: JudgeResults | None = None,
) -> list[str]:
    flags: list[str] = []

    if baseline.rouge_l - current.rouge_l > thresholds.rouge_l:
        flags.append(
            f"rouge_l regression: {current.rouge_l:.4f} vs baseline {baseline.rouge_l:.4f}"
        )

    if baseline.bertscore_f1 - current.bertscore_f1 > thresholds.bertscore_f1:
        flags.append(
            f"bertscore_f1 regression: {current.bertscore_f1:.4f} vs baseline {baseline.bertscore_f1:.4f}"
        )

    if current_judge and baseline_judge:
        if baseline_judge.clarity_mean - current_judge.clarity_mean > thresholds.judge_clarity:
            flags.append(
                f"judge_clarity regression: {current_judge.clarity_mean:.2f} vs baseline {baseline_judge.clarity_mean:.2f}"
            )
        if baseline_judge.safety_mean - current_judge.safety_mean > thresholds.judge_safety:
            flags.append(
                f"judge_safety regression: {current_judge.safety_mean:.2f} vs baseline {baseline_judge.safety_mean:.2f}"
            )

    return flags
