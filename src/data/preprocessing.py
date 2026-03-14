from __future__ import annotations

import re
from typing import Any

PROMPT_TEMPLATE = (
    "[INST] You are a medical assistant. Answer the following patient question clearly and safely.\n\n"
    "{input} [/INST] {output}"
)

_QUALITY_PATTERNS = re.compile(
    r"^(n/a|see (a )?doctor|see your doctor|consult (a |your )?doctor|not applicable)\.?$",
    re.IGNORECASE,
)


def format_prompt(record: dict[str, Any]) -> dict[str, Any]:
    text = PROMPT_TEMPLATE.format(input=record["input"], output=record["output"])
    return {**record, "text": text}


def passes_quality_filter(record: dict[str, Any]) -> bool:
    output = record.get("output", "").strip()
    return not bool(_QUALITY_PATTERNS.match(output))


def passes_length_filter(
    record: dict[str, Any],
    tokenizer: Any,
    min_input_tokens: int,
    min_output_tokens: int,
    max_length: int,
) -> bool:
    input_ids = tokenizer(record["input"], add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(record["output"], add_special_tokens=False)["input_ids"]
    if len(input_ids) < min_input_tokens or len(output_ids) < min_output_tokens:
        return False
    return len(input_ids) + len(output_ids) <= max_length


def compute_preprocessing_stats(
    initial: int,
    after_quality: int,
    after_length: int,
) -> dict[str, int]:
    return {
        "initial_count": initial,
        "removed_quality_filter": initial - after_quality,
        "removed_length_filter": after_quality - after_length,
        "final_count": after_length,
    }
