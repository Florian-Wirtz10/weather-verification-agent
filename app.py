"""Streamlit entry point.

Phase 1: synthetic precipitation fields, verification metrics, and plots,
wired end-to-end through the agent.run_verification orchestrator.
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import run_verification
from reports import save_report

load_dotenv()

st.set_page_config(page_title="Weather Verification Agent", layout="wide")
st.title("Weather Verification Agent")
st.caption("Phase 1: synthetic precipitation fields, categorical + continuous verification metrics.")

default_thresholds = os.getenv("DEFAULT_THRESHOLDS", "1,5,10")

with st.sidebar:
    st.header("Run settings")
    grid_size = st.number_input("Grid size (N x N)", min_value=8, max_value=256, value=64, step=8)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    thresholds_input = st.text_input("Precipitation thresholds (mm, comma-separated)", value=default_thresholds)
    run_clicked = st.button("Generate & verify", type="primary")

if run_clicked:
    thresholds = [float(t.strip()) for t in thresholds_input.split(",") if t.strip()]
    result = run_verification(shape=(grid_size, grid_size), seed=seed, thresholds=thresholds)
    st.session_state["result"] = result

result = st.session_state.get("result")

if result is None:
    st.info("Set parameters in the sidebar and click **Generate & verify** to run the pipeline.")
else:
    figures = result["figures"]
    col1, col2, col3 = st.columns(3)
    col1.pyplot(figures["forecast"])
    col2.pyplot(figures["obs"])
    col3.pyplot(figures["error"])

    metrics = result["metrics"]

    st.subheader("Continuous error metrics")
    st.table(pd.DataFrame([metrics["continuous"]]))

    st.subheader("Categorical scores by threshold")
    categorical_df = pd.DataFrame(metrics["categorical"]).T
    categorical_df.index.name = "threshold"
    st.table(categorical_df)

    if st.button("Save report to outputs/"):
        outdir = save_report(metrics, figures)
        st.success(f"Saved figures/tables/report to {outdir}/")
