from __future__ import annotations
import unsloth
import dataclasses
import logging
import random
from datetime import datetime, timezone
from typing import Optional

import torch
from datasets import Dataset

from src.evaluation.metrics.automated import compute_bertscore, compute_bleu, compute_rouge
from src.evaluation.metrics.latency import LatencyBenchmark
from src.evaluation.metrics.llm_judge import LLMJudge
from src.evaluation.regression import detect_regressions
from src.schemas import EvalConfig
from src.tracking.experiment_tracker import ExperimentTracker
from src.types import AutomatedMetricResults, EvalReport, EvalSample

logger = logging.getLogger(__name__)


def _load_model_and_tokenizer(model_path: str, max_seq_length: int) -> tuple:  # type: ignore[type-arg]
    from unsloth import FastLanguageModel

    # Unsloth auto-detects LoRA adapters (adapter_config.json) and loads
    # the base model + adapter in one call. 2x faster inference via for_inference().
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for batched generation: aligns RoPE positions
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _generate_responses(
    model: object,
    tokenizer: object,
    questions: list[str],
    batch_size: int,
) -> list[str]:
    from src.data.preprocessing import format_prompt_inference

    responses: list[str] = []
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i : i + batch_size]
        prompts = [format_prompt_inference(q, tokenizer) for q in batch_questions]

        inputs = tokenizer(  # type: ignore[call-arg]
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)  # type: ignore[union-attr]

        with torch.no_grad():
            outputs = model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,  # type: ignore[union-attr]
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True)  # type: ignore[union-attr]
            responses.append(text.strip())

        logger.info("Generated %d/%d responses", min(i + batch_size, len(questions)), len(questions))

    return responses


class EvalRunner:
    def __init__(
        self,
        model_path: str,
        config: EvalConfig,
        tracker: ExperimentTracker,
        run_id: str,
        baseline_run_name: str = "run_baseline",
    ):
        self.model_path = model_path
        self.config = config
        self.tracker = tracker
        self.run_id = run_id
        self.baseline_run_name = baseline_run_name

    def run(self, dataset: Dataset) -> EvalReport:
        model, tokenizer = _load_model_and_tokenizer(self.model_path, self.config.max_seq_length)

        rng = random.Random(self.config.seed)
        indices = list(range(len(dataset)))
        eval_indices = rng.sample(indices, min(self.config.num_samples, len(indices)))
        eval_subset = dataset.select(eval_indices)

        questions: list[str] = list(eval_subset["input"])
        references: list[str] = list(eval_subset["output"])

        responses = _generate_responses(model, tokenizer, questions, self.config.batch_size)

        automated = self.run_automated_metrics(responses, references)
        judge_results = self.run_llm_judge(questions, responses)
        latency_results = self.run_latency_benchmark()

        regression_flags = self._check_regressions(automated, judge_results)

        samples = self._build_samples(questions, references, responses, automated)
        report = EvalReport(
            run_id=self.run_id,
            model_path=self.model_path,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            num_samples=len(eval_subset),
            automated_metrics=automated,
            judge_results=judge_results,
            latency_results=latency_results,
            regression_flags=regression_flags,
        )

        self._log_to_tracker(report, samples)
        return report

    def run_automated_metrics(
        self, responses: list[str], references: list[str]
    ) -> AutomatedMetricResults:
        rouge = compute_rouge(responses, references)
        bertscore = compute_bertscore(responses, references)
        bleu = compute_bleu(responses, [[r] for r in references])
        return AutomatedMetricResults(
            rouge_1=rouge["rouge_1"],
            rouge_2=rouge["rouge_2"],
            rouge_l=rouge["rouge_l"],
            bertscore_f1=bertscore["bertscore_f1"],
            bleu_4=bleu["bleu_4"],
        )

    def run_llm_judge(
        self, questions: list[str], responses: list[str]
    ) -> Optional[object]:
        try:
            rng = random.Random(self.config.seed)
            n = min(self.config.judge_samples, len(questions))
            indices = rng.sample(range(len(questions)), n)
            sampled_q = [questions[i] for i in indices]
            sampled_r = [responses[i] for i in indices]
            judge = LLMJudge(model=self.config.judge_model)
            return judge.score_batch(sampled_q, sampled_r)
        except Exception as e:
            logger.warning("LLM judge eval failed: %s", e)
            return None

    def run_latency_benchmark(self) -> Optional[object]:
        try:
            benchmark = LatencyBenchmark(
                serving_url=self.config.serving_url,
                warmup_requests=self.config.latency_warmup_requests,
                benchmark_requests=self.config.latency_benchmark_requests,
            )
            return benchmark.run()
        except Exception as e:
            logger.warning("Latency benchmark failed (serving may not be running): %s", e)
            return None

    def _check_regressions(
        self,
        current: AutomatedMetricResults,
        current_judge: Optional[object],
    ) -> list[str]:
        try:
            baseline_metrics = self.tracker.get_run_metrics(self.baseline_run_name)
            if not baseline_metrics:
                return []
            baseline = AutomatedMetricResults(
                rouge_1=baseline_metrics.get("rouge_1", 0.0),
                rouge_2=baseline_metrics.get("rouge_2", 0.0),
                rouge_l=baseline_metrics.get("rouge_l", 0.0),
                bertscore_f1=baseline_metrics.get("bertscore_f1", 0.0),
                bleu_4=baseline_metrics.get("bleu_4", 0.0),
            )
            return detect_regressions(
                current, baseline, self.config.regression_thresholds
            )
        except Exception as e:
            logger.debug("Could not compare to baseline: %s", e)
            return []

    def _build_samples(
        self,
        questions: list[str],
        references: list[str],
        responses: list[str],
        automated: AutomatedMetricResults,
    ) -> list[EvalSample]:
        from src.evaluation.metrics.automated import compute_bertscore, compute_rouge
        samples = []
        for i, (q, ref, resp) in enumerate(zip(questions, references, responses)):
            rouge = compute_rouge([resp], [ref])
            bs = compute_bertscore([resp], [ref])
            samples.append(
                EvalSample(
                    sample_id=str(i),
                    question=q,
                    reference_answer=ref,
                    model_response=resp,
                    rouge_l=rouge["rouge_l"],
                    bertscore_f1=bs["bertscore_f1"],
                )
            )
        return samples

    def _log_to_tracker(self, report: EvalReport, samples: list[EvalSample]) -> None:
        metrics: dict[str, float] = {
            "rouge_1": report.automated_metrics.rouge_1,
            "rouge_2": report.automated_metrics.rouge_2,
            "rouge_l": report.automated_metrics.rouge_l,
            "bertscore_f1": report.automated_metrics.bertscore_f1,
            "bleu_4": report.automated_metrics.bleu_4,
        }

        if report.judge_results:
            jr = report.judge_results
            metrics.update({
                "judge_clarity_mean": jr.clarity_mean,
                "judge_clarity_std": jr.clarity_std,
                "judge_medical_accuracy_mean": jr.medical_accuracy_mean,
                "judge_medical_accuracy_std": jr.medical_accuracy_std,
                "judge_safety_mean": jr.safety_mean,
                "judge_safety_std": jr.safety_std,
            })

        if report.latency_results:
            lr = report.latency_results
            metrics.update({
                "latency_p50_ms": lr.p50_ms,
                "latency_p95_ms": lr.p95_ms,
                "latency_p99_ms": lr.p99_ms,
                "latency_throughput_rps": lr.throughput_rps,
            })

        self.tracker.log_metrics(metrics)

        regression_detected = len(report.regression_flags) > 0
        self.tracker.set_tag("regression_detected", str(regression_detected).lower())

        if regression_detected:
            for flag in report.regression_flags:
                logger.warning("REGRESSION: %s", flag)

        report_dict = dataclasses.asdict(report)
        self.tracker.log_dict(report_dict, "eval_report.json")

        rng = random.Random(42)
        sample_indices = rng.sample(range(len(samples)), min(20, len(samples)))
        sample_outputs = [dataclasses.asdict(samples[i]) for i in sample_indices]
        self.tracker.log_dict({"samples": sample_outputs}, "sample_outputs.json")
