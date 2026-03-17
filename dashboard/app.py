"""Streamlit dashboard: live demo + experiment comparison."""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SERVING_URL = os.environ.get("SERVING_URL", "http://localhost:8000")
RUNS_DIR = Path(os.environ.get("RUNS_DIR", "runs"))

METRIC_COLS = [
    "rouge_l", "rouge_1", "rouge_2", "bertscore_f1", "bleu_4",
    "judge_clarity_mean", "judge_medical_accuracy_mean", "judge_safety_mean",
]

st.set_page_config(page_title="LLM Eval Harness", layout="wide")

page = st.sidebar.radio("Navigate", ["Live Demo", "Experiment Comparison"])


def _get_runs() -> pd.DataFrame:
    rows = []
    for summary_path in sorted(RUNS_DIR.glob("*/metrics_summary.json")):
        run_name = summary_path.parent.name
        metrics = json.loads(summary_path.read_text())
        rows.append({"run": run_name, **metrics})
    return pd.DataFrame(rows)


def _get_loss_history(run_name: str) -> pd.DataFrame:
    run_dir = RUNS_DIR / run_name
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    if "train/loss" not in ea.Tags().get("scalars", []):
        return pd.DataFrame()
    events = ea.Scalars("train/loss")
    return pd.DataFrame({"step": [e.step for e in events], "loss": [e.value for e in events]})


def _call_generate(question: str) -> dict:  # type: ignore[type-arg]
    resp = httpx.post(
        f"{SERVING_URL}/generate",
        json={"question": question, "max_new_tokens": 256},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


if page == "Live Demo":
    st.title("Medical Q&A — Live Demo")
    st.caption(f"Inference via {SERVING_URL}")

    run_names = sorted([p.parent.name for p in RUNS_DIR.glob("*/metrics_summary.json")])

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area(
            "Patient question", height=100,
            placeholder="e.g. I have had a headache for 3 days. What should I do?",
        )
    with col2:
        st.selectbox("Model checkpoint", run_names or ["run_lora_r16"])

    if st.button("Generate response", type="primary") and question:
        with st.spinner("Generating..."):
            try:
                result = _call_generate(question)
                st.subheader("Response")
                st.write(result["response"])
                st.caption(f"Latency: {result['latency_ms']:.1f}ms | Tokens: {result['tokens_generated']}")
            except Exception as e:
                st.error(f"Serving error: {e}")

elif page == "Experiment Comparison":
    st.title("Experiment Comparison")

    df = _get_runs()
    if df.empty:
        st.warning(f"No completed runs found under '{RUNS_DIR}/'. Run experiments first.")
        st.stop()

    present_metric_cols = [c for c in METRIC_COLS if c in df.columns]

    st.subheader("Metrics Table")
    display_df = df[["run"] + present_metric_cols].set_index("run")
    st.dataframe(display_df, use_container_width=True)

    if present_metric_cols:
        st.subheader("Metric Comparison")
        selected_metric = st.selectbox("Metric", present_metric_cols)
        chart_df = df[["run", selected_metric]].dropna()
        fig = px.bar(chart_df, x="run", y=selected_metric, color="run", title=selected_metric)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Training Loss Curves")
    st.caption("Reads TensorBoard event files from runs/. Launch `tensorboard --logdir runs/` for the full interactive view.")
    run_names = df["run"].tolist()
    selected_runs = st.multiselect("Select runs", run_names, default=run_names[:3])
    loss_data = []
    for run_name in selected_runs:
        try:
            run_loss_df = _get_loss_history(run_name)
            if not run_loss_df.empty:
                run_loss_df["run"] = run_name
                loss_data.append(run_loss_df)
        except Exception:
            pass
    if loss_data:
        fig2 = px.line(pd.concat(loss_data), x="step", y="loss", color="run", title="Training Loss")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No training loss data found. Loss curves are available in TensorBoard.")

    st.subheader("Sample Outputs Browser")
    run_for_samples = st.selectbox("Run", run_names, key="samples_run")
    if run_for_samples and st.button("Load samples"):
        samples_path = RUNS_DIR / run_for_samples / "sample_outputs.json"
        if samples_path.exists():
            samples = json.loads(samples_path.read_text())
            for s in samples.get("samples", [])[:5]:
                with st.expander(f"Sample {s['sample_id']}"):
                    st.write("**Question:**", s["question"])
                    st.write("**Reference:**", s["reference_answer"])
                    st.write("**Response:**", s["model_response"])
                    st.caption(f"ROUGE-L: {s['rouge_l']:.3f}")
        else:
            st.error(f"sample_outputs.json not found for run '{run_for_samples}'")
