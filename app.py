"""Streamlit entry point.

Phase 1: synthetic precipitation fields, verification metrics, and plots,
wired end-to-end through the agent.run_verification orchestrator.

Phase 2 adds a second tab: real-case reflectivity verification (风雷
REFMOSAIC forecasts vs radar CREF observations) driven by a natural-language
request that DeepSeek parses into a choice of thresholds and which figures
(from a fixed menu) to show — it never generates plotting code.
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import PLOT_MENU, parse_request, run_fenglei_case_verification, run_verification, select_figures
from reports import save_report

load_dotenv()

st.set_page_config(page_title="Weather Verification Agent", layout="wide")
st.title("Weather Verification Agent")

tab_synthetic, tab_real_case = st.tabs(["Phase 1: 合成数据 demo", "Phase 2: 真实个例检验"])

with tab_synthetic:
    st.caption("Phase 1: synthetic precipitation fields, categorical + continuous verification metrics.")

    default_thresholds = os.getenv("DEFAULT_THRESHOLDS", "1,5,10")

    with st.sidebar:
        st.header("Phase 1 run settings")
        grid_size = st.number_input("Grid size (N x N)", min_value=8, max_value=256, value=64, step=8)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        thresholds_input = st.text_input(
            "Precipitation thresholds (mm, comma-separated)", value=default_thresholds
        )
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

with tab_real_case:
    st.caption(
        "风雷 REFMOSAIC 反射率预报 vs 雷达 CREF 观测 — 2025-04-11 湖北大范围强对流个例。"
        "external/ 下的样例数据不随仓库分发，请先在 .env 中配置 FENGLEI_CASE_DIR / RADAR_OBS_DIR。"
    )

    with st.sidebar:
        st.header("Phase 2 个例设置")
        fenglei_dir = st.text_input("风雷 REFMOSAIC 目录", value=os.getenv("FENGLEI_CASE_DIR", ""))
        radar_dir = st.text_input("雷达 CREF 观测目录", value=os.getenv("RADAR_OBS_DIR", ""))
        lat_min = st.number_input("lat_min", value=28.0)
        lat_max = st.number_input("lat_max", value=35.0)
        lon_min = st.number_input("lon_min", value=108.0)
        lon_max = st.number_input("lon_max", value=118.0)

    nl_request = st.text_area(
        "用自然语言描述你想看什么(DeepSeek 只会从固定的图表菜单里选图，不会生成代码)",
        value="检验一下这次个例，给我 CSI/POD/FAR 随预报时效变化的曲线，还有首尾两帧的反射率场对比。",
    )
    run_case_clicked = st.button("解析请求并运行检验", type="primary")

    if run_case_clicked:
        try:
            parsed = parse_request(nl_request)
        except Exception as exc:  # DeepSeek unavailable/misconfigured — fall back to defaults
            st.warning(f"自然语言解析失败，使用默认参数（全部图表 + 默认阈值）。原因: {exc}")
            parsed = {"thresholds": [10.0, 20.0, 35.0], "plots": list(PLOT_MENU), "notes": ""}

        st.session_state["fenglei_parsed"] = parsed
        try:
            case_result = run_fenglei_case_verification(
                fenglei_dir=fenglei_dir,
                radar_dir=radar_dir,
                bbox=(lat_min, lat_max, lon_min, lon_max),
                thresholds=tuple(parsed["thresholds"]),
            )
            st.session_state["fenglei_result"] = case_result
        except Exception as exc:
            st.session_state["fenglei_result"] = None
            st.error(f"检验流程运行失败: {exc}")

    parsed = st.session_state.get("fenglei_parsed")
    case_result = st.session_state.get("fenglei_result")

    if parsed is not None:
        st.write(
            f"**解析结果** — 阈值: {parsed['thresholds']} dBZ；图表: {parsed['plots']}"
            + (f"；说明: {parsed['notes']}" if parsed.get("notes") else "")
        )

    if case_result is None:
        st.info("填好目录/区域参数，用自然语言描述需求，然后点击上方按钮运行。")
    else:
        selected_figures = select_figures(case_result["figures"], parsed["plots"])
        for name, fig in selected_figures.items():
            st.pyplot(fig)

        st.subheader("各预报时效分类检验指标")
        rows = []
        for entry in case_result["per_lead"]:
            for threshold, scores in entry["categorical"].items():
                rows.append({"lead_minutes": entry["lead_minutes"], "threshold": threshold, **scores})
        st.dataframe(pd.DataFrame(rows), width="stretch")
