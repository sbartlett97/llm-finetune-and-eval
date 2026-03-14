"""Streamlit dashboard: live demo + experiment comparison."""
from __future__ import annotations

import json
import os

import httpx
import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st

SERVING_URL = os.environ.get("SERVING_URL", "http://localhost:8000")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
EXPERIMENT_NAME = "llm-eval-harness"

mlflow.set_tracking_uri(MLFLOW_URI)

st.set_page_config(page_title="LLM Eval Harness", layout="wide")

page = st.sidebar.radio("Navigate", ["Live Demo", "Experiment Comparison"])


def _get_runs() -> pd.DataFrame:
    try:
        df = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
        df.columns = [c.replace("metrics.", "").replace("params.", "p_").replace("tags.", "tag_") for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _call_generate(question: str, run_id: str) -> dict:  # type: ignore[type-arg]
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

    runs_df = _get_runs()
    run_names = runs_df["tag_mlflow.runName"].dropna().unique().tolist() if not runs_df.empty else []

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area("Patient question", height=100,
                                placeholder="e.g. I have had a headache for 3 days. What should I do?")
    with col2:
        selected_run = st.selectbox("Model checkpoint", run_names or ["run_lora_r16"])

    if st.button("Generate response", type="primary") and question:
        with st.spinner("Generating..."):
            try:
                result = _call_generate(question, selected_run)
                st.subheader("Response")
                st.write(result["response"])
                st.caption(f"Latency: {result['latency_ms']:.1f}ms | Tokens: {result['tokens_generated']}")
            except Exception as e:
                st.error(f"Serving error: {e}")

elif page == "Experiment Comparison":
    st.title("Experiment Comparison")

    df = _get_runs()
    if df.empty:
        st.warning("No MLflow runs found. Run experiments first.")
        st.stop()

    metric_cols = [c for c in df.columns if c in [
        "rouge_l", "rouge_1", "rouge_2", "bertscore_f1", "bleu_4",
        "judge_clarity_mean", "judge_medical_accuracy_mean", "judge_safety_mean",
    ]]
    display_cols = ["tag_mlflow.runName"] + metric_cols
    display_df = df[[c for c in display_cols if c in df.columns]].rename(
        columns={"tag_mlflow.runName": "run"}
    )

    st.subheader("Metrics Table")
    st.dataframe(display_df.set_index("run"), use_container_width=True)

    if metric_cols:
        st.subheader("Metric Comparison")
        selected_metric = st.selectbox("Metric", metric_cols, index=0)
        chart_df = display_df[["run", selected_metric]].dropna()
        fig = px.bar(chart_df, x="run", y=selected_metric, color="run", title=selected_metric)
        st.plotly_chart(fig, use_container_width=True)

    if "tag_mlflow.runName" in df.columns:
        st.subheader("Training Loss Curves")
        run_names = df["tag_mlflow.runName"].dropna().unique().tolist()
        selected_runs = st.multiselect("Select runs", run_names, default=run_names[:3])

        loss_data = []
        client = mlflow.tracking.MlflowClient()
        for run_name in selected_runs:
            run_rows = df[df["tag_mlflow.runName"] == run_name]
            if run_rows.empty:
                continue
            run_id = run_rows.iloc[0]["run_id"]
            history = client.get_metric_history(run_id, "loss")
            for point in history:
                loss_data.append({"run": run_name, "step": point.step, "loss": point.value})

        if loss_data:
            loss_df = pd.DataFrame(loss_data)
            fig2 = px.line(loss_df, x="step", y="loss", color="run", title="Training Loss")
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sample Outputs Browser")
    run_for_samples = st.selectbox("Run", df["tag_mlflow.runName"].dropna().unique().tolist(), key="samples_run")
    if run_for_samples and st.button("Load samples"):
        run_rows = df[df["tag_mlflow.runName"] == run_for_samples]
        if not run_rows.empty:
            run_id = run_rows.iloc[0]["run_id"]
            try:
                artifact_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="sample_outputs.json"
                )
                samples = json.loads(open(artifact_path).read())
                for s in samples.get("samples", [])[:5]:
                    with st.expander(f"Sample {s['sample_id']}"):
                        st.write("**Question:**", s["question"])
                        st.write("**Reference:**", s["reference_answer"])
                        st.write("**Response:**", s["model_response"])
                        st.caption(f"ROUGE-L: {s['rouge_l']:.3f}")
            except Exception as e:
                st.error(f"Could not load samples: {e}")
