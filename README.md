# LLM Eval Harness — Medical Q&A Fine-tuning & Evaluation

A production-grade LLM fine-tuning and evaluation harness built as a portfolio project. The system fine-tunes **Qwen2.5-3B-Instruct (`unsloth/Qwen2.5-3B-Instruct`)** on a medical dialogue dataset using LoRA/QLoRA and provides a structured, repeatable evaluation framework across multiple metric dimensions.

The emphasis is on the *system around the model*: experiment tracking, metric design, regression detection, and production serving — not just the model weights.

---

## Results

> **Fill in after running the experiment matrix.**

### Benchmark Results

| Run | ROUGE-L | BERTScore F1 | Judge: Clarity | Judge: Accuracy | Judge: Safety | p95 Latency |
|---|---|---|---|---|---|---|
| `run_baseline` | — | — | — | — | — | — |
| `run_lora_r8` | — | — | — | — | — | — |
| `run_lora_r16` | — | — | — | — | — | — |
| `run_lora_r32` | — | — | — | — | — | — |
| `run_lora_r16_2ep` | — | — | — | — | — | — |
| **Target** | ≥ 0.30 | ≥ 0.87 | ≥ 4.0/5 | ≥ 3.8/5 | ≥ 4.2/5 | < 200ms |

### Key Findings

> _Fill in after analysis. Example prompts:_
> - Which LoRA rank gave the best ROUGE-L / BERTScore tradeoff?
> - Did additional epochs (r16 vs r16_2ep) help or overfit?
> - What was the biggest driver of judge score improvement?
> - Were any regression flags triggered?

### Training Costs

> _Fill in after runs. Example:_

| Run | GPU | Duration | Cost |
|---|---|---|---|
| `run_lora_r8` | A100 40GB | — | — |
| `run_lora_r16` | A100 40GB | — | — |
| `run_lora_r32` | A100 40GB | — | — |
| `run_lora_r16_2ep` | A100 40GB | — | — |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                        │
│                                                                 │
│  HuggingFace Dataset → Preprocessing → LoRA Fine-tuning        │
│                                  ↓                              │
│                         Model Checkpoints                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Eval Runner                            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Automated   │  │ LLM-as-judge │  │ Latency benchmark    │  │
│  │  metrics     │  │  (LangChain) │  │ (FastAPI endpoint)   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         └─────────────────┴─────────────────────┘              │
│                               ↓                                 │
│                TensorBoard (+ JSON sidecars)                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Serving Layer                            │
│                                                                 │
│           FastAPI inference endpoint (Dockerised)              │
│                         ↓                                       │
│                  Streamlit demo UI                              │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Config-driven.** All training and eval runs are controlled by YAML config files — no magic numbers in code.
- **Decoupled eval.** The eval runner is independent of training. It accepts any HuggingFace-compatible model path.
- **Tracker abstraction.** All TensorBoard calls go through `ExperimentTracker` — the training and eval code is tracker-agnostic.
- **Fail loudly.** Eval failures raise exceptions and are flagged in the TensorBoard JSON sidecar. No silent zero scores.
- **Reproducible.** Given the same config YAML and seed, any run produces identical results. Dataset splits are fixed and saved as JSON sidecars alongside TensorBoard events.

For full rationale on each decision, see [`DECISIONS.md`](DECISIONS.md) *(to be written)*.

---

## Project Structure

```
eval-harness/
├── configs/
│   ├── data_config.yaml
│   ├── eval_config.yaml
│   └── training/
│       ├── lora_r8.yaml
│       ├── lora_r16.yaml          # default experiment
│       ├── lora_r32.yaml
│       └── lora_r16_2ep.yaml
│
├── src/
│   ├── data/                      # dataset loading, preprocessing, splits
│   ├── training/                  # LoRA fine-tuner, callbacks
│   ├── evaluation/
│   │   ├── eval_runner.py         # centrepiece — runs full metric suite
│   │   ├── metrics/
│   │   │   ├── automated.py       # ROUGE, BERTScore, BLEU
│   │   │   ├── llm_judge.py       # LangChain + structured output scoring
│   │   │   └── latency.py         # p50/p95/p99 benchmark
│   │   └── regression.py          # flags degradation vs baseline
│   ├── tracking/                  # ExperimentTracker (TensorBoard abstraction)
│   └── serving/                   # FastAPI app + model loader
│
├── dashboard/app.py               # Streamlit: live demo + experiment comparison
├── scripts/
│   ├── train.py                   # entry point: train one run
│   ├── eval.py                    # entry point: eval one run
│   └── run_all_experiments.sh     # runs full experiment matrix
│
├── tests/
│   ├── unit/                      # 26 tests, run without GPU
│   └── integration/               # requires datasets package
│
├── reports/                       # eval_report.json per run, committed to repo
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── notebooks/results_analysis.ipynb
└── PRD.md                         # full product requirements and component specs
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- `HF_TOKEN` — HuggingFace token (for `unsloth/Qwen2.5-3B-Instruct`)
- `OPENAI_API_KEY` — for LLM-as-judge (GPT-4o-mini)
- CUDA GPU for training and meaningful latency benchmarks

### Install

```bash
git clone <repo-url>
cd eval-harness

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # CUDA 13 (local dev)
# pip install -r requirements-cu121.txt  # CUDA 12.x / Docker

cp .env.example .env
# Edit .env and fill in HF_TOKEN and OPENAI_API_KEY
```

### Run tests (no GPU required)

```bash
pip install -r requirements-dev.txt
pytest
```

### Train a single run

```bash
python scripts/train.py --config configs/training/lora_r16.yaml
```

### Evaluate a run

```bash
python scripts/eval.py --run-id run_lora_r16
```

### Run the full experiment matrix

```bash
bash scripts/run_all_experiments.sh
```

> This runs all five experiments sequentially. Expect ~1–2 hours per training run on an A100 40GB.

### View results in TensorBoard

```bash
tensorboard --logdir runs/
# Open http://localhost:6006
```

Each run is written to `runs/{run_name}/`. After training completes you can compare loss curves, eval metrics, and hyperparameters across all runs in the TensorBoard UI.

### Start the serving stack

```bash
# Inference endpoint only
docker compose --profile serving up

# Full stack: FastAPI + MLflow server + Streamlit dashboard
MODEL_RUN_ID=run_lora_r16 docker compose --profile full up
```

The FastAPI endpoint will be at `http://localhost:8000`. Health check: `GET /health`.

### Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

---

## Experiment Matrix

| Run ID | Description | LoRA Rank | Epochs | Notes |
|---|---|---|---|---|
| `run_baseline` | Base Qwen2.5-3B-Instruct, no fine-tuning | — | — | Upper bound for degradation detection |
| `run_lora_r8` | LoRA rank 8 | 8 | 1 | Minimal adapter capacity |
| `run_lora_r16` | LoRA rank 16 *(default)* | 16 | 1 | Primary experiment |
| `run_lora_r32` | LoRA rank 32 | 32 | 1 | Higher capacity, more VRAM |
| `run_lora_r16_2ep` | Rank 16, 2 epochs | 16 | 2 | Tests multi-epoch tradeoff |

All runs use Qwen2.5-3B-Instruct (`unsloth/Qwen2.5-3B-Instruct`) with 4-bit QLoRA, `lora_alpha=2×r`, `target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, `max_seq_length=2048`.

---

## Metrics

### Automated (full 5,000-sample test set)

| Metric | Library | What it measures |
|---|---|---|
| ROUGE-L | `rouge_score` | Longest common subsequence overlap with reference |
| ROUGE-1, ROUGE-2 | `rouge_score` | Unigram and bigram overlap |
| BERTScore F1 | `bert_score` | Semantic similarity via contextual embeddings |
| BLEU-4 | `sacrebleu` | N-gram precision (secondary signal) |

### LLM-as-Judge (100-sample subset, GPT-4o-mini)

Each response is scored 1–5 on three dimensions by the judge model:

- **Clarity** — is it clearly written for a non-expert?
- **Medical accuracy** — does it appear medically sound?
- **Safety** — does it appropriately recommend professional consultation?

Mean and standard deviation are logged per run. The judge prompt uses structured output parsing via LangChain to ensure JSON validity.

### Latency (live FastAPI endpoint)

100 sequential requests after a 10-request warmup. Metrics: p50, p95, p99 (ms), throughput (req/s).

---

## Regression Detection

After each eval run, all metrics are compared against `run_baseline`. If any metric degrades beyond its configured threshold, the run is flagged `regression_detected: true` in the TensorBoard JSON sidecar and a warning is printed.

Thresholds (configurable in `configs/eval_config.yaml`):

| Metric | Max allowed drop |
|---|---|
| ROUGE-L | 0.02 |
| BERTScore F1 | 0.01 |
| Judge: Clarity | 0.3 |
| Judge: Safety | 0.2 |

---

## Serving API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/ready` | Readiness — confirms model is loaded |
| `POST` | `/generate` | Inference endpoint |

**Request:**
```json
{
  "question": "I have had a headache for 3 days. What should I do?",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "response": "...",
  "model_run_id": "run_lora_r16",
  "latency_ms": 143.2,
  "tokens_generated": 87
}
```

---

## Environment Variables

```bash
# Required
HF_TOKEN=                    # HuggingFace token (for unsloth/Qwen2.5-3B-Instruct)
OPENAI_API_KEY=              # GPT-4o-mini for LLM judge

# Optional
MODEL_RUN_ID=run_lora_r16    # Which checkpoint the server loads (resolves to checkpoints/{run_id}/)
MODEL_PATH=                  # Override: explicit local path to model weights
MAX_SEQ_LENGTH=2048          # Context window for inference serving (default: 2048)
LOG_LEVEL=INFO
```

See `.env.example` for a full template.

---

## Tech Stack

| Component | Technology |
|---|---|
| Base model | Qwen2.5-3B-Instruct (`unsloth/Qwen2.5-3B-Instruct`) |
| Fine-tuning | `unsloth` (`FastLanguageModel`) + `trl` (SFTTrainer) |
| Quantisation | Unsloth 4-bit QLoRA (NF4, `load_in_4bit=True`) |
| Automated metrics | `rouge_score`, `bert_score`, `sacrebleu` |
| LLM judge | `langchain` + `langchain-openai` |
| Experiment tracking | TensorBoard 2.x |
| Inference serving | FastAPI + Uvicorn |
| Containerisation | Docker + docker-compose |
| Demo UI | Streamlit + Plotly |
| Testing | pytest + pytest-cov |
| Linting / types | ruff + mypy (strict) |

---

## Reproducing a Run

Any run is fully reproducible from its config file and seed:

```bash
python scripts/train.py --config configs/training/lora_r16.yaml
python scripts/eval.py --run-id run_lora_r16
```

The dataset split indices are saved as JSON sidecars alongside TensorBoard events for each run.

---

## Out of Scope (v1)

The following are natural v2 extensions:

- Multi-GPU / distributed training
- RLHF or DPO alignment (LoRA SFT only in v1)
- Automated hyperparameter search (Optuna, Ray Tune)
- CI/CD pipeline (GitHub Actions)
- Real-time production monitoring (Prometheus, Grafana)
- CPU-optimised inference (GGUF/GGML)
- Concurrency/load testing on the serving layer

---

## Notes & Open Questions

> _Fill in as experiments run._
>
> - [ ] Verify `lavita/ChatDoctor-HealthCareMagic-100k` licence permits public portfolio use.
> - [ ] Decide: commit LoRA adapter weights via Git LFS, or link to HuggingFace Hub?
> - [ ] Run pilot comparison of GPT-4o-mini vs Claude Haiku as judge on 20 samples.
> - [ ] Decide: deploy Streamlit dashboard to Streamlit Cloud for live demo?

---

*Built by Samuel Bartlett · Full specification: [PRD.md](PRD.md)*