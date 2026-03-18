from __future__ import annotations

from pydantic import BaseModel, Field


class LoraConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainingConfig(BaseModel):
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    seed: int = 42
    max_seq_length: int = 1024
    dataloader_num_workers: int = 4
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"


class RunConfig(BaseModel):
    run_name: str
    base_model: str = "unsloth/SmolLM3-3B-128K"
    output_dir: str
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


class DataConfig(BaseModel):
    dataset_name: str = "lavita/ChatDoctor-HealthCareMagic-100k"
    train_size: int = 90000
    val_size: int = 5000
    test_size: int = 5000
    seed: int = 42
    max_length: int = 4096
    train_max_length: int = 1024
    min_input_tokens: int = 10
    min_output_tokens: int = 20


class RegressionThresholds(BaseModel):
    rouge_l: float = 0.02
    bertscore_f1: float = 0.01
    judge_clarity: float = 0.3
    judge_safety: float = 0.2


class EvalConfig(BaseModel):
    batch_size: int = 16
    num_samples: int = 500
    judge_model: str = "gpt-4o-mini"
    judge_samples: int = 100
    latency_warmup_requests: int = 10
    latency_benchmark_requests: int = 100
    seed: int = 42
    serving_url: str = "http://localhost:8000"
    regression_thresholds: RegressionThresholds = Field(default_factory=RegressionThresholds)


class GenerateRequest(BaseModel):
    question: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


class GenerateResponse(BaseModel):
    response: str
    model_run_id: str
    latency_ms: float
    tokens_generated: int
