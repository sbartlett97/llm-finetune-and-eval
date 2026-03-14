from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AutomatedMetricResults:
    rouge_1: float
    rouge_2: float
    rouge_l: float
    bertscore_f1: float
    bleu_4: float


@dataclass
class JudgeResults:
    clarity_mean: float
    clarity_std: float
    medical_accuracy_mean: float
    medical_accuracy_std: float
    safety_mean: float
    safety_std: float
    num_samples: int


@dataclass
class LatencyResults:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    num_requests: int


@dataclass
class EvalSample:
    sample_id: str
    question: str
    reference_answer: str
    model_response: str
    rouge_l: float
    bertscore_f1: float
    judge_clarity: Optional[int] = None
    judge_medical_accuracy: Optional[int] = None
    judge_safety: Optional[int] = None
    judge_reasoning: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class EvalReport:
    run_id: str
    model_path: str
    timestamp: str
    num_samples: int
    automated_metrics: AutomatedMetricResults
    judge_results: Optional[JudgeResults] = None
    latency_results: Optional[LatencyResults] = None
    regression_flags: list[str] = field(default_factory=list)


@dataclass
class TrainingResult:
    run_name: str
    output_dir: str
    final_train_loss: float
    final_val_loss: float
    mlflow_run_id: str
