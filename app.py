
from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from models import evaluate_models_for_single_day, live_forecast_for_single_day
from utils import (
    DEFAULT_APP_DIR,
    create_daep_forecast_only_plot,
    create_daep_forecast_plot,
    create_spread_diagnostic_plot,
    create_forecast_comparison_plot,
    create_rt_forecast_only_plot,
    create_rt_forecast_plot,
    get_available_run_days,
    load_feature_store,
    load_proxy_feature_store,
)

st.set_page_config(page_title="BGE Runtime-Constrained Price Forecasting", page_icon="⚡", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1220px; padding-top: 1.1rem; padding-bottom: 1rem;}
.hero {padding: 1rem 1.2rem 0.5rem 0.2rem;}
.hero h1 {margin: 0; color: #1d2a44; font-size: 3rem; font-weight: 800;}
.hero p {margin: 0.35rem 0 0 0; color: #5d6b82; font-size: 1.05rem;}
.card {background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%); border: 1px solid #e7eef9; border-radius: 18px; padding: 1rem 1.1rem; box-shadow: 0 8px 24px rgba(20, 40, 90, 0.07);}
.metric-box {background: linear-gradient(135deg, #edf4ff 0%, #ffffff 100%); border: 1px solid #dbe7ff; border-radius: 16px; padding: 0.9rem 1rem; box-shadow: 0 6px 18px rgba(31, 71, 136, 0.08);}
.metric-label {font-size: 0.9rem; color: #62718a; margin-bottom: 0.2rem;}
.metric-value {font-size: 1.65rem; font-weight: 800; color: #1d2a44;}
div[data-testid="stTabs"] button {font-size: 1rem;}
</style>
""", unsafe_allow_html=True)

MODEL_OPTIONS = ["RandomForest", "ExtraTrees", "XGBoost", "Voting", "Stacking"]

defaults = {
    "app_dir": str(DEFAULT_APP_DIR),
    "selected_models": ["RandomForest", "ExtraTrees", "Stacking"],
    "train_window": 14,
    "live_source": "Use saved latest snapshot",
    "eval_result": None,
    "live_result": None,
    "eval_day": None,
    "detail_model_eval": None,
    "spread_model_eval": None,
    "updater_status": None,
    "proxy_mae_source": "Saved historical features",
    "refresh_proxy_cache": False,
    "proxy_window_days": 7,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v



def run_live_updater(app_dir: str, run_day: pd.Timestamp, refresh_proxy_cache: bool = False, proxy_window_days: int = 7, pjm_key: str = ""):
    script = Path(__file__).with_name("update_live_data.py")
    if not script.exists():
        return False, "update_live_data.py was not found beside app.py."
    cmd = [
        sys.executable,
        str(script),
        "--app-dir",
        str(app_dir),
        "--run-day",
        pd.Timestamp(run_day).strftime("%Y-%m-%d"),
        "--proxy-days",
        str(int(proxy_window_days)),
    ]
    if refresh_proxy_cache:
        cmd.append("--refresh-proxy-cache")
    env = os.environ.copy()
    if pjm_key.strip():
        env["PJM_API_KEY"] = pjm_key.strip()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    msg = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode == 0:
        return True, "Live data update completed successfully. Fresh PJM, load-forecast, and weather inputs are available."
    combo = "\n".join([x for x in [msg, err] if x]).strip()
    return False, combo or f"Live updater failed with exit code {proc.returncode}."

@st.cache_data(show_spinner=False)
def load_app_data(app_dir: str):
    da = load_feature_store(Path(app_dir), "DA_total")
    rt = load_feature_store(Path(app_dir), "RT_total")
    days = get_available_run_days(da["data"], rt["data"])
    return da, rt, days

def nice_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["DA MAE", "DA RMSE", "RT MAE", "RT RMSE"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:,.2f}")
    return out


def render_glossary():
    with st.expander("Glossary", expanded=False):
        glossary = pd.DataFrame(
            [
                ["DA", "Day-Ahead locational marginal price for the next operating day."],
                ["RT", "Real-Time locational marginal price reflecting shorter-term balancing conditions."],
                ["DAEP", "Day-Ahead Energy Price."],
                ["Runtime", "The information cutoff used to prepare the next-day forecast, aligned to the 10:00 AM workflow."],
                ["Historical Proxy MAE", "A recent reference error used to compare model performance before an experimental live run."],
                ["Dynamic Proxy Cache", "A compact overwrite-only cache rebuilt from the latest refreshed history for recent proxy-error calculations."],
                ["Load Forecast", "Forecasted system load used as a forward-looking predictor when a fresh next-day panel is available."],
                ["CDH / HDH", "Cooling Degree Hours and Heating Degree Hours derived from temperature to capture weather-driven demand effects."],
                ["Forecast Spread", "The forecast Day-Ahead price minus the forecast Real-Time price for the same delivery hour."],
            ],
            columns=["Term", "Meaning"],
        )
        st.dataframe(glossary, use_container_width=True, hide_index=True)

def render_faq():
    with st.expander("FAQ", expanded=False):
        items = [
            ("What is the difference between Model Evaluation and Live Forecast?",
             "Model Evaluation checks model performance on a selected historical day with known outcomes. Live Forecast uses the latest available inputs to generate an experimental next-day forecast."),
            ("Does the workflow respect the 10:00 AM runtime?",
             "Yes. The workflow is aligned to the 10:00 AM information cutoff and is designed to avoid using future realized values."),
            ("What is Historical Proxy MAE?",
             "It is a recent reference error used to compare models before an experimental live forecast. It can use the stable historical feature store or the optional dynamic proxy cache."),
            ("Can this app be used for financial or market decisions?",
             "No. This is an interactive research demonstration. Forecasts and diagnostics are provided for research, visualization, and reproducibility rather than financial or market decision-making."),
            ("What is the dynamic proxy cache?",
             "It is a separate overwrite-only cache that stores only the most recent selected window rebuilt from refreshed history."),
            ("Does refreshing the proxy cache change the main historical feature store?",
             "No. The proxy cache is separate and is used only for the proxy-error calculation when selected."),
            ("What is Stacking?",
             "Stacking is an ensemble model that combines multiple base learners into one final prediction model."),
            ("Which models are used inside Stacking?",
             "The current demonstration uses RandomForest, ExtraTrees, and XGBoost as base learners, with HuberRegressor as the meta-model."),
            ("How should I interpret the DA–RT comparison?",
             "It is a forecast comparison only. The spread is Day-Ahead minus Real-Time forecast price and should not be interpreted as a trading recommendation."),
        ]
        for q, answer in items:
            st.markdown(f"**{q}**")
            st.write(answer)


def render_footer():
    st.markdown(
        """
        <div style="margin-top:2rem;border-top:1px solid #e7eef9;padding-top:0.9rem;text-align:center;color:#5d6b82;font-size:0.95rem;">
            <b>Created by Chibuike C. Ibebuchi</b>
        </div>
        <div style="text-align:center;color:#7a8699;font-size:0.85rem;margin-top:0.35rem;">
            Interactive research demonstration only. Not intended for financial or market decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("## Settings")
    st.session_state.train_window = st.selectbox("Training window (run-days)", [7, 14], index=[7, 14].index(st.session_state.train_window))
    st.session_state.selected_models = st.multiselect("Models", MODEL_OPTIONS, default=st.session_state.selected_models)
    st.session_state.live_source = st.radio("Live data source", ["Use saved latest snapshot", "Fetch latest PJM/weather data now"], index=0 if st.session_state.live_source.startswith("Use") else 1)
    if st.session_state.live_source == "Use saved latest snapshot":
        st.warning("Saved snapshot mode is for research demonstration only and does not represent a freshly refreshed live run.")
    else:
        st.info("Experimental live refresh. A PJM API key is required for fresh PJM and weather inputs.")
    session_pjm_key = st.text_input(
        "PJM API key",
        value="",
        type="password",
        help="Required only for live refresh. Evaluation works without a key. The app does not save your key.",
    )
    st.caption("Live refresh requires a PJM API key. The key is used only for this session and is not saved.")
    st.markdown("### Proxy MAE")
    st.session_state.proxy_mae_source = st.radio(
        "Proxy MAE source",
        ["Saved historical features", "Dynamic proxy cache"],
        index=0 if st.session_state.proxy_mae_source == "Saved historical features" else 1,
    )
    st.session_state.proxy_window_days = st.selectbox(
        "Dynamic proxy cache window",
        [7, 14],
        index=[7, 14].index(st.session_state.proxy_window_days),
        help="Used only when you refresh the dynamic proxy cache.",
    )
    st.session_state.refresh_proxy_cache = st.checkbox(
        "Refresh dynamic proxy cache from latest live downloads (overwrite)",
        value=st.session_state.refresh_proxy_cache,
        help="Overwrites the existing proxy cache with only the last N days from the latest refreshed history.",
    )

st.markdown('<div class="hero"><h1>⚡ BGE Runtime-Constrained Price Forecasting</h1><p>Interactive research demonstration of Day-Ahead and Real-Time electricity price forecasts.</p></div>', unsafe_allow_html=True)
render_glossary()
render_faq()

try:
    da_store, rt_store, run_days = load_app_data(st.session_state.app_dir)
except Exception as e:
    st.error(f"Could not load app data: {e}")
    st.stop()

if not run_days:
    st.error("No common run-days found in the DA and RT feature stores.")
    st.stop()

if st.session_state.eval_day is None:
    st.session_state.eval_day = run_days[-1]

eval_tab, live_tab = st.tabs(["📊 Model Evaluation", "🚀 Live Forecast"])

with eval_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Evaluation")
    st.caption("Select one historical day. The app trains on the prior 7 or 14 run-days, then evaluates only that chosen day.")

    c1, c2, c3 = st.columns([1.4, 1, 1.2])
    with c1:
        st.session_state.eval_day = st.selectbox(
            "Evaluation date",
            options=run_days,
            index=run_days.index(st.session_state.eval_day) if st.session_state.eval_day in run_days else len(run_days) - 1,
            format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
            key="eval_day_select",
        )
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Training window</div><div class="metric-value">{st.session_state.train_window} days</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Models selected</div><div class="metric-value">{len(st.session_state.selected_models)}</div></div>', unsafe_allow_html=True)

    if st.button("Run model evaluation", type="primary", key="run_eval_btn"):
        if not st.session_state.selected_models:
            st.warning("Select at least one model.")
        else:
            with st.spinner("Running evaluation..."):
                st.session_state.eval_result = evaluate_models_for_single_day(
                    da_store, rt_store, pd.Timestamp(st.session_state.eval_day), st.session_state.selected_models, st.session_state.train_window
                )
            summary = st.session_state.eval_result["summary"]
            if not summary.empty:
                reference_model = "Stacking" if "Stacking" in summary["Model"].tolist() else summary.iloc[0]["Model"]
                st.session_state.spread_model_eval = reference_model
                st.session_state.detail_model_eval = reference_model

    if st.session_state.eval_result is not None:
        result = st.session_state.eval_result
        summary = result["summary"]
        preds = result["predictions"]
        spread_details = result["spread_details"]

        reference_model = "Stacking" if "Stacking" in summary["Model"].tolist() else summary.iloc[0]["Model"]
        reference_row = summary.loc[summary["Model"] == reference_model].iloc[0]
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Reference model", reference_model)
        mc2.metric("DA MAE", f'{reference_row["DA MAE"]:.2f}')
        mc3.metric("RT MAE", f'{reference_row["RT MAE"]:.2f}')

        st.markdown("### Model scorecard")
        st.dataframe(nice_scorecard(summary), use_container_width=True, hide_index=True)

        t1, t2, t3, t4 = st.tabs(["Day-Ahead • all models", "Real-Time • all models", "DA–RT spread", "Detailed model view"])
        with t1:
            st.plotly_chart(create_forecast_comparison_plot(preds, "DA Total"), use_container_width=True, key="eval_da_all")
            st.plotly_chart(create_daep_forecast_plot(preds, reference_model), use_container_width=True, key=f"eval_da_reference_{reference_model}")
        with t2:
            st.plotly_chart(create_forecast_comparison_plot(preds, "RT Total"), use_container_width=True, key="eval_rt_all")
            st.plotly_chart(create_rt_forecast_plot(preds, reference_model), use_container_width=True, key=f"eval_rt_reference_{reference_model}")
        with t3:
            model_list = summary["Model"].tolist()
            if st.session_state.spread_model_eval not in model_list:
                st.session_state.spread_model_eval = model_list[0]
            st.session_state.spread_model_eval = st.selectbox("Spread diagnostic model", model_list, key="spread_model_eval_select")
            spread_model = st.session_state.spread_model_eval
            spread_df = spread_details[spread_model].copy()
            st.plotly_chart(create_spread_diagnostic_plot(spread_df, spread_model), use_container_width=True, key=f"spread_plot_{spread_model}")
            st.dataframe(
                spread_df[["Timestamp", "DA Actual", "RT Actual", "DA Forecast", "RT Forecast", "Actual Spread (DA - RT)", "Forecast Spread (DA - RT)", "Absolute Spread Error"]],
                use_container_width=True,
                hide_index=True,
            )
        with t4:
            model_list = summary["Model"].tolist()
            if st.session_state.detail_model_eval not in model_list:
                st.session_state.detail_model_eval = model_list[0]
            st.session_state.detail_model_eval = st.selectbox("Detailed model", model_list, key="detail_model_eval_select")
            view_model = st.session_state.detail_model_eval
            d1, d2 = st.columns(2)
            with d1:
                st.plotly_chart(create_daep_forecast_plot(preds, view_model), use_container_width=True, key=f"detail_da_{view_model}")
            with d2:
                st.plotly_chart(create_rt_forecast_plot(preds, view_model), use_container_width=True, key=f"detail_rt_{view_model}")
    st.markdown('</div>', unsafe_allow_html=True)

with live_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Forecast")
    today = pd.Timestamp(date.today()).normalize()
    st.caption("Experimental live mode uses today's date, trains on prior run-days, and produces next-day forecasts for the selected models.")
    l1, l2, l3 = st.columns([1,1,1.2])
    l1.markdown(f'<div class="metric-box"><div class="metric-label">Live forecast date</div><div class="metric-value">{today.strftime("%Y-%m-%d")}</div></div>', unsafe_allow_html=True)
    l2.markdown(f'<div class="metric-box"><div class="metric-label">Training window</div><div class="metric-value">{st.session_state.train_window} days</div></div>', unsafe_allow_html=True)
    l3.markdown(f'<div class="metric-box"><div class="metric-label">Source</div><div class="metric-value" style="font-size:1.1rem;">{"Saved snapshot" if st.session_state.live_source.startswith("Use") else "Fresh PJM/API"}</div></div>', unsafe_allow_html=True)

    if st.button("Run live forecast", type="primary", key="run_live_btn"):
        if not st.session_state.selected_models:
            st.warning("Select at least one model.")
        else:
            source_for_forecast = "Use saved latest snapshot"
            st.session_state.updater_status = None

            if st.session_state.live_source == "Fetch latest PJM/weather data now":
                if not session_pjm_key.strip():
                    st.session_state.updater_status = ("warning", "Live refresh requires a PJM API key. Enter the key in the sidebar password field; the app does not save it.")
                else:
                    with st.spinner("Updating local live data before forecasting..."):
                        ok, msg = run_live_updater(
                            st.session_state.app_dir,
                            today,
                            refresh_proxy_cache=bool(st.session_state.refresh_proxy_cache),
                            proxy_window_days=int(st.session_state.proxy_window_days),
                            pjm_key=session_pjm_key,
                        )
                        st.session_state.updater_status = ("success" if ok else "warning", msg)
                    source_for_forecast = "Use latest local live update"

            load_app_data.clear()
            da_loaded, rt_loaded, _ = load_app_data(st.session_state.app_dir)
            proxy_da_store = None
            proxy_rt_store = None
            if st.session_state.proxy_mae_source == "Dynamic proxy cache":
                try:
                    proxy_da_store = load_proxy_feature_store(Path(st.session_state.app_dir), "DA_total")
                    proxy_rt_store = load_proxy_feature_store(Path(st.session_state.app_dir), "RT_total")
                except Exception:
                    proxy_da_store = None
                    proxy_rt_store = None

            with st.spinner("Running live forecast..."):
                st.session_state.live_result = live_forecast_for_single_day(
                    Path(st.session_state.app_dir), da_loaded, rt_loaded, today,
                    st.session_state.selected_models, st.session_state.train_window, source_for_forecast,
                    proxy_da_store=proxy_da_store, proxy_rt_store=proxy_rt_store
                )

    if st.session_state.live_result is not None:
        if st.session_state.updater_status is not None:
            level, msg = st.session_state.updater_status
            if level == "success":
                st.success(msg)
            else:
                st.warning(msg)
        live = st.session_state.live_result
        if st.session_state.live_source == "Use saved latest snapshot":
            st.warning("Snapshot mode is for research demonstration only and does not represent a freshly refreshed live run.")
        else:
            st.info("Experimental live mode is using freshly refreshed PJM and weather inputs.")
        st.success(live["Status Message"])
        live_preds = live["predictions"]
        live_summary = live["summary"]
        show = live_summary.copy()
        show["Historical Proxy MAE"] = show["Historical Proxy MAE"].map(lambda x: f"{x:,.2f}")
        st.dataframe(show, use_container_width=True, hide_index=True)

        x1, x2, x3 = st.tabs(["Day-Ahead forecast", "Real-Time forecast", "DA–RT comparison"])
        with x1:
            st.plotly_chart(create_daep_forecast_only_plot(live_preds), use_container_width=True, key="live_da_all")
        with x2:
            st.plotly_chart(create_rt_forecast_only_plot(live_preds), use_container_width=True, key="live_rt_all")
        with x3:
            st.caption(f"DA–RT comparison shown for the reference model: {live['Reference Model']}.")
            st.dataframe(live["comparison"], use_container_width=True, hide_index=True)
            bar = px.bar(
                live["comparison"].assign(Hour=lambda d: pd.to_datetime(d["Timestamp"]).dt.strftime("%H:%M")),
                x="Hour",
                y=["Forecast DA", "Forecast RT"],
                barmode="group",
                height=360,
                title="Next-day DA and RT forecast comparison by hour",
            )
            bar.update_layout(margin=dict(l=18, r=18, t=55, b=18))
            st.plotly_chart(bar, use_container_width=True, key="live_signal_bar")
    st.markdown('</div>', unsafe_allow_html=True)


render_footer()
