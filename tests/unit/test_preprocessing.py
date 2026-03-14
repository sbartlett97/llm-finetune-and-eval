from __future__ import annotations

import pytest

from src.data.preprocessing import (
    PROMPT_TEMPLATE,
    compute_preprocessing_stats,
    format_prompt,
    passes_quality_filter,
)


def test_format_prompt_contains_input_and_output() -> None:
    record = {"input": "My question", "output": "Doctor answer"}
    result = format_prompt(record)
    assert "My question" in result["text"]
    assert "Doctor answer" in result["text"]
    assert "[INST]" in result["text"]
    assert "[/INST]" in result["text"]


def test_format_prompt_preserves_original_fields() -> None:
    record = {"input": "q", "output": "a", "extra": "value"}
    result = format_prompt(record)
    assert result["extra"] == "value"


@pytest.mark.parametrize("output", [
    "N/A",
    "n/a",
    "see doctor",
    "See a doctor",
    "Consult a doctor.",
    "See your doctor",
    "not applicable",
])
def test_passes_quality_filter_rejects_boilerplate(output: str) -> None:
    assert passes_quality_filter({"output": output}) is False


def test_passes_quality_filter_accepts_real_response() -> None:
    record = {"output": "You should rest and drink plenty of fluids. If symptoms persist, see a doctor."}
    assert passes_quality_filter(record) is True


def test_passes_quality_filter_empty_output() -> None:
    assert passes_quality_filter({"output": ""}) is True


def test_compute_preprocessing_stats() -> None:
    stats = compute_preprocessing_stats(1000, 950, 900)
    assert stats["initial_count"] == 1000
    assert stats["removed_quality_filter"] == 50
    assert stats["removed_length_filter"] == 50
    assert stats["final_count"] == 900


def test_prompt_template_has_placeholders() -> None:
    assert "{input}" in PROMPT_TEMPLATE
    assert "{output}" in PROMPT_TEMPLATE
