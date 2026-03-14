from __future__ import annotations

import pytest

from src.evaluation.metrics.automated import compute_bleu, compute_rouge


def test_compute_rouge_identical() -> None:
    preds = ["the cat sat on the mat"]
    refs = ["the cat sat on the mat"]
    result = compute_rouge(preds, refs)
    assert result["rouge_l"] == pytest.approx(1.0, abs=0.01)
    assert result["rouge_1"] == pytest.approx(1.0, abs=0.01)


def test_compute_rouge_different() -> None:
    preds = ["completely different words here"]
    refs = ["the cat sat on the mat"]
    result = compute_rouge(preds, refs)
    assert result["rouge_l"] < 0.3


def test_compute_rouge_batch() -> None:
    preds = ["the cat sat", "dog ran fast"]
    refs = ["the cat sat on the mat", "the dog ran very fast"]
    result = compute_rouge(preds, refs)
    assert 0.0 <= result["rouge_l"] <= 1.0
    assert 0.0 <= result["rouge_1"] <= 1.0
    assert 0.0 <= result["rouge_2"] <= 1.0


def test_compute_bleu_identical() -> None:
    preds = ["the cat sat on the mat"]
    refs = [["the cat sat on the mat"]]
    result = compute_bleu(preds, refs)
    assert result["bleu_4"] == pytest.approx(1.0, abs=0.01)


def test_compute_bleu_different() -> None:
    preds = ["completely different text"]
    refs = [["the cat sat on the mat"]]
    result = compute_bleu(preds, refs)
    assert result["bleu_4"] < 0.1


def test_compute_bleu_takes_list_of_lists_for_references() -> None:
    preds = ["hello world"]
    refs = [["hello world", "hi world"]]
    result = compute_bleu(preds, refs)
    assert 0.0 <= result["bleu_4"] <= 1.0
