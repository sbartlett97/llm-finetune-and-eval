from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.eval_runner import EvalRunner
from src.schemas import EvalConfig
from src.types import AutomatedMetricResults


@pytest.fixture()
def eval_config() -> EvalConfig:
    return EvalConfig(
        batch_size=2,
        num_samples=4,
        judge_samples=2,
        latency_warmup_requests=1,
        latency_benchmark_requests=2,
    )


@pytest.fixture()
def mock_tracker() -> MagicMock:
    tracker = MagicMock()
    tracker.current_run_id = "test-run-id"
    tracker.get_run_metrics.side_effect = ValueError("no baseline")
    return tracker


@patch("src.evaluation.eval_runner._load_model_and_tokenizer")
@patch("src.evaluation.eval_runner._generate_responses")
def test_eval_runner_run_returns_report(
    mock_generate: MagicMock,
    mock_load: MagicMock,
    eval_config: EvalConfig,
    mock_tracker: MagicMock,
) -> None:
    mock_load.return_value = (MagicMock(), MagicMock())
    mock_generate.return_value = [
        "You should rest and drink fluids.",
        "See a doctor if it persists.",
        "Take over the counter medication.",
        "Monitor your symptoms carefully.",
    ]

    from datasets import Dataset
    dataset = Dataset.from_dict({
        "input": ["q1", "q2", "q3", "q4"],
        "output": ["ref1", "ref2", "ref3", "ref4"],
    })

    runner = EvalRunner(
        model_path="./fake_model",
        config=eval_config,
        tracker=mock_tracker,
        run_id="test_run",
    )

    with (
        patch.object(runner, "run_llm_judge", return_value=None),
        patch.object(runner, "run_latency_benchmark", return_value=None),
    ):
        report = runner.run(dataset)

    assert report.run_id == "test_run"
    assert report.num_samples <= 4
    assert isinstance(report.automated_metrics, AutomatedMetricResults)
    assert 0.0 <= report.automated_metrics.rouge_l <= 1.0
    mock_tracker.log_metrics.assert_called()
    mock_tracker.set_tag.assert_called_with("regression_detected", "false")


def test_regression_detection_no_baseline(
    eval_config: EvalConfig, mock_tracker: MagicMock
) -> None:
    runner = EvalRunner(
        model_path="./fake",
        config=eval_config,
        tracker=mock_tracker,
        run_id="test",
    )
    current = AutomatedMetricResults(
        rouge_1=0.3, rouge_2=0.1, rouge_l=0.25, bertscore_f1=0.85, bleu_4=0.1
    )
    flags = runner._check_regressions(current, None)
    assert flags == []
