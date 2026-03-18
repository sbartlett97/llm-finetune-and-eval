from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data.preprocessing import (
    compute_preprocessing_stats,
    format_prompt,
    format_prompt_inference,
    passes_quality_filter,
)


def _make_tokenizer(return_value: str = "<|im_start|>...") -> MagicMock:
    tok = MagicMock()
    tok.apply_chat_template.return_value = return_value
    return tok


def test_format_prompt_calls_apply_chat_template() -> None:
    tok = _make_tokenizer("<|im_start|>user\nMy question<|im_end|>\n<|im_start|>assistant\nDoctor answer<|im_end|>")
    record = {"input": "My question", "output": "Doctor answer"}
    result = format_prompt(record, tok)

    tok.apply_chat_template.assert_called_once()
    call_kwargs = tok.apply_chat_template.call_args
    messages = call_kwargs[0][0]
    assert any(m["role"] == "user" and "My question" in m["content"] for m in messages)
    assert any(m["role"] == "assistant" and "Doctor answer" in m["content"] for m in messages)
    assert call_kwargs[1].get("add_generation_prompt") is False
    assert "text" in result


def test_format_prompt_preserves_original_fields() -> None:
    tok = _make_tokenizer()
    record = {"input": "q", "output": "a", "extra": "value"}
    result = format_prompt(record, tok)
    assert result["extra"] == "value"


def test_format_prompt_inference_uses_add_generation_prompt() -> None:
    tok = _make_tokenizer("<|im_start|>user\nheadache<|im_end|>\n<|im_start|>assistant\n")
    result = format_prompt_inference("headache 3 days", tok)

    tok.apply_chat_template.assert_called_once()
    call_kwargs = tok.apply_chat_template.call_args
    messages = call_kwargs[0][0]
    assert not any(m["role"] == "assistant" for m in messages)
    assert call_kwargs[1].get("add_generation_prompt") is True
    assert isinstance(result, str)


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
