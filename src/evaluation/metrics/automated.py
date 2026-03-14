from __future__ import annotations

import logging
import warnings

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

logger = logging.getLogger(__name__)


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)
    return {
        "rouge_1": sum(scores["rouge1"]) / len(scores["rouge1"]),
        "rouge_2": sum(scores["rouge2"]) / len(scores["rouge2"]),
        "rouge_l": sum(scores["rougeL"]) / len(scores["rougeL"]),
    }


def compute_bertscore(predictions: list[str], references: list[str]) -> dict[str, float]:
    import torch
    from bert_score import score as bert_score_fn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("BERTScore running on CPU — this will be slow for large datasets")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, f1 = bert_score_fn(
            predictions,
            references,
            lang="en",
            device=device,
            batch_size=64,
            verbose=False,
        )
    return {"bertscore_f1": float(f1.mean())}


def compute_bleu(predictions: list[str], references: list[list[str]]) -> dict[str, float]:
    """Compute BLEU-4. `references` must be List[List[str]] — one list of refs per hypothesis."""
    bleu = BLEU(max_ngram_order=4)
    result = bleu.corpus_score(predictions, references)
    return {"bleu_4": result.score / 100.0}
