from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.metrics.llm_judge import JudgeScore, LLMJudge, _JUDGE_PROMPT
from src.types import JudgeResults


def test_judge_prompt_contains_required_placeholders() -> None:
    assert "{question}" in _JUDGE_PROMPT
    assert "{response}" in _JUDGE_PROMPT


def test_judge_prompt_mentions_all_dimensions() -> None:
    assert "clarity" in _JUDGE_PROMPT
    assert "medical_accuracy" in _JUDGE_PROMPT
    assert "safety" in _JUDGE_PROMPT


def test_judge_score_model_validates_range() -> None:
    score = JudgeScore(clarity=4, medical_accuracy=3, safety=5, reasoning="Good response.")
    assert score.clarity == 4
    assert score.medical_accuracy == 3
    assert score.safety == 5


@patch("src.evaluation.metrics.llm_judge.ChatOpenAI")
def test_llm_judge_score_single_success(mock_chat: MagicMock) -> None:
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = JudgeScore(
        clarity=4, medical_accuracy=3, safety=5, reasoning="Clear and safe."
    )
    mock_chat.return_value.with_structured_output.return_value = mock_chain

    judge = LLMJudge(model="gpt-4o-mini")
    result = judge.score_single("What is a headache?", "A headache is pain in the head.")
    assert result is not None
    assert result.clarity == 4
    assert result.safety == 5


@patch("src.evaluation.metrics.llm_judge.ChatOpenAI")
def test_llm_judge_score_single_handles_failure(mock_chat: MagicMock) -> None:
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("API error")
    mock_chat.return_value.with_structured_output.return_value = mock_chain

    judge = LLMJudge(model="gpt-4o-mini")
    result = judge.score_single("question", "response")
    assert result is None


@patch("src.evaluation.metrics.llm_judge.ChatOpenAI")
def test_llm_judge_score_batch_aggregates_correctly(mock_chat: MagicMock) -> None:
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [
        JudgeScore(clarity=4, medical_accuracy=3, safety=5, reasoning="r1"),
        JudgeScore(clarity=2, medical_accuracy=4, safety=3, reasoning="r2"),
    ]
    mock_chat.return_value.with_structured_output.return_value = mock_chain

    judge = LLMJudge(model="gpt-4o-mini")
    result = judge.score_batch(["q1", "q2"], ["r1", "r2"])

    assert isinstance(result, JudgeResults)
    assert result.clarity_mean == pytest.approx(3.0)
    assert result.safety_mean == pytest.approx(4.0)
    assert result.num_samples == 2


@patch("src.evaluation.metrics.llm_judge.ChatOpenAI")
def test_llm_judge_raises_if_all_fail(mock_chat: MagicMock) -> None:
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("API error")
    mock_chat.return_value.with_structured_output.return_value = mock_chain

    judge = LLMJudge(model="gpt-4o-mini")
    with pytest.raises(RuntimeError, match="All judge evaluations failed"):
        judge.score_batch(["q"], ["r"])
