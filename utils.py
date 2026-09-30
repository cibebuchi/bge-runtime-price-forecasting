
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

DEFAULT_APP_DIR = Path(__file__).resolve().parent
TIME_COL = "datetime_beginning_ept"
RUNTIME_HOUR = 10
HORIZON_HOURS = 24
SNAPSHOT_LAGS_H = [1, 2, 3, 6, 12, 24, 48, 72, 168]
SNAPSHOT_ROLL_WINDOWS_H = [6, 24, 72, 168]
FORECAST_COL = "load_forecast_mw"
FORECAST_PRESENT_COL = "load_forecast_present"
BASE_URL = "https://api.pjm.com/api/v1"

def _json_load(path: Path):
    import json
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def load_feature_store(app_dir: Path, label: str):
    feat_dir = Path(app_dir) / "03_features"
    df = pd.read_parquet(feat_dir / f"{label}__ENGINEERED.parquet")
    feature_cols = _json_load(feat_dir / f"{label}__feature_cols.json")
    target_col = "total_da" if label == "DA_total" else "total_rt"
    df["run_day"] = pd.to_datetime(df["run_day"]); df["run_time"] = pd.to_datetime(df["run_time"]); df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])
    return {"label": label, "target_col": target_col, "feature_cols": feature_cols, "data": df.sort_values(["run_day","target_timestamp"]).reset_index(drop=True)}

def load_proxy_feature_store(app_dir: Path, label: str):
    feat_dir = Path(app_dir) / "07_proxy_mae"
    df = pd.read_parquet(feat_dir / f"{label}__ENGINEERED.parquet")
    feature_cols = _json_load(feat_dir / f"{label}__feature_cols.json")
    target_col = "total_da" if label == "DA_total" else "total_rt"
    df["run_day"] = pd.to_datetime(df["run_day"]); df["run_time"] = pd.to_datetime(df["run_time"]); df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])
    return {"label": label, "target_col": target_col, "feature_cols": feature_cols, "data": df.sort_values(["run_day","target_timestamp"]).reset_index(drop=True)}

def get_available_run_days(da_df, rt_df):
    da_days = set(pd.to_datetime(da_df["run_day"]).dt.normalize().unique()); rt_days = set(pd.to_datetime(rt_df["run_day"]).dt.normalize().unique())
    return [pd.Timestamp(d) for d in sorted(da_days.intersection(rt_days))]

def latest_common_run_day(da_df, rt_df):
    days = get_available_run_days(da_df, rt_df)
    if not days: raise RuntimeError("No common run-days found.")
    return days[-1]

def build_single_day_train_test(df, target_col, feature_cols, eval_day, train_window_days):
    d = df.copy(); d["run_day"] = pd.to_datetime(d["run_day"]).dt.normalize(); eval_day = pd.Timestamp(eval_day).normalize()
    train_start = eval_day - pd.Timedelta(days=int(train_window_days))
    train = d[(d["run_day"] >= train_start) & (d["run_day"] < eval_day)].copy()
    test = d[d["run_day"] == eval_day].copy()
    if train.empty: raise RuntimeError("No training rows available for that date and window.")
    if test.empty: raise RuntimeError("No test rows exist for that selected day.")
    X_train = train[feature_cols].apply(pd.to_numeric, errors="coerce"); y_train = pd.to_numeric(train[target_col], errors="coerce").to_numpy(dtype=float)
    X_test = test[feature_cols].apply(pd.to_numeric, errors="coerce"); y_test = pd.to_numeric(test[target_col], errors="coerce").to_numpy(dtype=float)
    meta = test[["run_day","run_time","target_timestamp"]].copy().rename(columns={"target_timestamp":"Timestamp"})
    return X_train, y_train, X_test, y_test, meta

def apply_business_layout(fig, title, ytitle="Price (USD/MWh)", height=430):
    fig.update_layout(title=title, template="plotly_white", height=height, margin=dict(l=18, r=18, t=55, b=18), legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0, bgcolor="rgba(255,255,255,0.85)"), xaxis_title="Time", yaxis_title=ytitle)
    return fig

def create_forecast_comparison_plot(preds_df, target):
    target_key = {"DA Total":"DA_total", "RT Total":"RT_total"}.get(target, target)
    sub = preds_df[preds_df["Target"] == target_key].copy().sort_values(["Model","Timestamp"])
    actual = sub[["Timestamp","Actual"]].dropna().drop_duplicates("Timestamp").sort_values("Timestamp")
    fig = go.Figure()
    if not actual.empty:
        fig.add_trace(go.Scatter(x=actual["Timestamp"], y=actual["Actual"], mode="lines+markers", name="Actual", line=dict(width=3)))
    for model_name, part in sub.groupby("Model"):
        fig.add_trace(go.Scatter(x=part["Timestamp"], y=part["Prediction"], mode="lines", name=model_name, line=dict(width=2), opacity=0.92))
    title = "Day-Ahead prices • all models" if target_key == "DA_total" else "Real-Time prices • all models"
    return apply_business_layout(fig, title)

def create_daep_forecast_plot(preds_df, model_name):
    sub = preds_df[(preds_df["Target"] == "DA_total") & (preds_df["Model"] == model_name)].copy().sort_values("Timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["Timestamp"], y=sub["Actual"], mode="lines+markers", name="DAEP Actual", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=sub["Timestamp"], y=sub["Prediction"], mode="lines+markers", name=f"Forecast ({model_name})", line=dict(width=2, dash="dash")))
    return apply_business_layout(fig, f"Day-Ahead forecast vs actual • {model_name}")

def create_rt_forecast_plot(preds_df, model_name):
    sub = preds_df[(preds_df["Target"] == "RT_total") & (preds_df["Model"] == model_name)].copy().sort_values("Timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["Timestamp"], y=sub["Actual"], mode="lines+markers", name="RT Actual", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=sub["Timestamp"], y=sub["Prediction"], mode="lines+markers", name=f"Forecast ({model_name})", line=dict(width=2, dash="dash")))
    return apply_business_layout(fig, f"Real-Time forecast vs actual • {model_name}")

def create_daep_forecast_only_plot(preds_df):
    sub = preds_df[preds_df["Target"] == "DA Total"].copy().sort_values(["Model","Timestamp"])
    fig = go.Figure()
    for model_name, part in sub.groupby("Model"):
        fig.add_trace(go.Scatter(x=part["Timestamp"], y=part["Prediction"], mode="lines+markers", name=model_name, line=dict(width=2)))
    return apply_business_layout(fig, "Next-day Day-Ahead forecast • all models")

def create_rt_forecast_only_plot(preds_df):
    sub = preds_df[preds_df["Target"] == "RT Total"].copy().sort_values(["Model","Timestamp"])
    fig = go.Figure()
    for model_name, part in sub.groupby("Model"):
        fig.add_trace(go.Scatter(x=part["Timestamp"], y=part["Prediction"], mode="lines+markers", name=model_name, line=dict(width=2)))
    return apply_business_layout(fig, "Next-day Real-Time forecast • all models")

def create_spread_diagnostic_plot(spread_df, model_name):
    d = spread_df.copy().sort_values("Timestamp")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Timestamp"], y=d["Actual Spread (DA - RT)"], name="Actual spread", opacity=0.55))
    fig.add_trace(go.Scatter(x=d["Timestamp"], y=d["Forecast Spread (DA - RT)"], mode="lines+markers", name="Forecast spread", line=dict(width=3)))
    return apply_business_layout(fig, f"DA–RT spread diagnostic • {model_name}", ytitle="Spread (USD/MWh)", height=430)

def _safe_last_row_leq(df, t, cols):
    sub = df.loc[:t, cols]
    if len(sub) == 0: return pd.Series([np.nan]*len(cols), index=cols)
    return sub.iloc[-1]

def add_cyclical_time_features(df, ts_col):
    d = df.copy(); t = pd.to_datetime(d[ts_col]); hour=t.dt.hour; dow=t.dt.dayofweek; doy=t.dt.dayofyear; month=t.dt.month
    d["is_weekend"]=(dow>=5).astype(int); d["hour_sin"]=np.sin(2*np.pi*hour/24.0); d["hour_cos"]=np.cos(2*np.pi*hour/24.0); d["dow_sin"]=np.sin(2*np.pi*dow/7.0); d["dow_cos"]=np.cos(2*np.pi*dow/7.0); d["doy_sin"]=np.sin(2*np.pi*doy/365.25); d["doy_cos"]=np.cos(2*np.pi*doy/365.25); d["month_sin"]=np.sin(2*np.pi*month/12.0); d["month_cos"]=np.cos(2*np.pi*month/12.0)
    return d

def build_snapshot_feature_row(df, runtime, base_cols):
    feat={}; last = df.loc[runtime, base_cols] if runtime in df.index else _safe_last_row_leq(df, runtime, base_cols)
    for c in base_cols: feat[f"{c}_at_runtime"] = float(last[c]) if pd.notna(last[c]) else np.nan
    for lag in SNAPSHOT_LAGS_H:
        t_lag = runtime - pd.Timedelta(hours=lag); row = df.loc[t_lag, base_cols] if t_lag in df.index else _safe_last_row_leq(df, t_lag, base_cols)
        for c in base_cols:
            feat[f"{c}_lag{lag}h"] = float(row[c]) if pd.notna(row[c]) else np.nan
            a = feat.get(f"{c}_at_runtime", np.nan); b = feat.get(f"{c}_lag{lag}h", np.nan)
            feat[f"{c}_delta{lag}h"] = (a - b) if (np.isfinite(a) and np.isfinite(b)) else np.nan
    for w in SNAPSHOT_ROLL_WINDOWS_H:
        w0 = runtime - pd.Timedelta(hours=w - 1); win = df.loc[w0:runtime, base_cols]
        for c in base_cols:
            s = win[c]; feat[f"{c}_rollmean{w}h"] = float(s.mean()) if s.notna().any() else np.nan; feat[f"{c}_rollstd{w}h"] = float(s.std()) if s.notna().any() else np.nan; feat[f"{c}_rollmin{w}h"] = float(s.min()) if s.notna().any() else np.nan; feat[f"{c}_rollmax{w}h"] = float(s.max()) if s.notna().any() else np.nan
    return feat

def load_curated_merged(app_dir):
    p = Path(app_dir) / "02_curated" / "bge_weather_load_merged.parquet"
    df = pd.read_parquet(p); df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last").reset_index(drop=True)

def load_live_override_merged(app_dir):
    p = Path(app_dir) / "06_live" / "bge_weather_load_merged_live.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if "datetime_beginning_ept" in df.columns:
            df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"])
        return df
    return None

def load_live_override_forecast_panel(app_dir):
    p = Path(app_dir) / "06_live" / "load_forecast_runtime10am_next48h_live.parquet"
    if p.exists():
        f = pd.read_parquet(p)
        for c in ["runtime_hour_ept", "forecast_hour_beginning_ept", "evaluated_at_ept", "_run_time"]:
            if c in f.columns:
                f[c] = pd.to_datetime(f[c], errors="coerce")
        return f
    return None

def load_saved_forecast_panel(app_dir):
    p = Path(app_dir) / "01_raw_parquet" / "load_forecast_runtime10am_next48h.parquet"
    f = pd.read_parquet(p)
    for c in ["runtime_hour_ept","forecast_hour_beginning_ept","evaluated_at_ept","_run_time"]:
        if c in f.columns: f[c] = pd.to_datetime(f[c], errors="coerce")
    return f

def forecast_vec_for_run(fcst, runtime, target_index):
    runtime = pd.Timestamp(runtime).floor("h"); ti = pd.to_datetime(target_index).floor("h")
    if "_run_time" in fcst.columns and fcst["_run_time"].notna().any():
        sub = fcst.loc[fcst["_run_time"].dt.floor("h") == runtime, ["forecast_hour_beginning_ept","forecast_load_mw"]].copy()
        if not sub.empty:
            m = sub.set_index("forecast_hour_beginning_ept")["forecast_load_mw"]; return np.array([m.get(t, np.nan) for t in ti], dtype=float)
    if "runtime_hour_ept" in fcst.columns:
        for off in [0,-1,-2,1]:
            rk = (runtime + pd.Timedelta(hours=off)).floor("h")
            sub = fcst.loc[fcst["runtime_hour_ept"].dt.floor("h") == rk, ["forecast_hour_beginning_ept","forecast_load_mw"]].copy()
            if not sub.empty:
                m = sub.set_index("forecast_hour_beginning_ept")["forecast_load_mw"]; vec = np.array([m.get(t, np.nan) for t in ti], dtype=float)
                if np.isfinite(vec).sum() > 0: return vec
    return np.full(len(ti), np.nan, dtype=float)

def build_live_block_from_merged(merged_df, fcst, feature_cols, live_day):
    df = merged_df.copy(); df[TIME_COL] = pd.to_datetime(df[TIME_COL]); df = df.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last").set_index(TIME_COL)
    runtime = pd.Timestamp(live_day).normalize() + pd.Timedelta(hours=RUNTIME_HOUR)
    target_start = pd.Timestamp(live_day).normalize() + pd.Timedelta(days=1)
    target_index = pd.date_range(target_start, periods=HORIZON_HOURS, freq="h")
    base_cols_master = ["total_rt","energy_rt","congestion_rt","loss_rt","total_da","energy_da","congestion_da","loss_da","n_nodes_rt","n_nodes_da","temp","rhum","wspd","prcp","pres","cdh","hdh","cdh2","hdh2","load_actual_mw","gas_price","outage_mw"]
    base_cols = [c for c in base_cols_master if c in df.columns]
    snap = build_snapshot_feature_row(df, runtime, base_cols)
    X = pd.DataFrame([snap] * len(target_index))
    X["Target Timestamp"] = target_index; X["Run Time"] = runtime; X["Run Day"] = pd.Timestamp(live_day).normalize()
    if fcst is None or not isinstance(fcst, pd.DataFrame) or fcst.empty:
        X[FORECAST_COL] = np.nan
        X[FORECAST_PRESENT_COL] = 0
    else:
        X[FORECAST_COL] = forecast_vec_for_run(fcst, runtime, target_index)
        X[FORECAST_PRESENT_COL] = np.isfinite(X[FORECAST_COL]).astype(np.int8)
    X = add_cyclical_time_features(X, "Target Timestamp")
    for c in feature_cols:
        if c not in X.columns: X[c] = np.nan
    return X[["Run Day","Run Time","Target Timestamp"] + feature_cols].copy()

def load_pjm_key():
    env_key = os.environ.get("PJM_API_KEY", "").strip()
    if env_key:
        return env_key
    raise RuntimeError("PJM API key not found. Enter a key in the app before using live refresh.")

def call_pjm(endpoint, params):
    headers = {"Ocp-Apim-Subscription-Key": load_pjm_key()}
    r = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def fetch_window(endpoint, params):
    params = dict(params); params["startRow"] = 1; params["rowCount"] = 50000; all_items=[]
    while True:
        data = call_pjm(endpoint, params); items = data.get("items", []); total = data.get("totalRows", 0)
        if not items: break
        all_items.extend(items); params["startRow"] += len(items)
        if params["startRow"] > total: break
    return pd.DataFrame(all_items)

def fetch_bge_hourly_lmps(run_day, market):
    start = (pd.Timestamp(run_day).normalize() - pd.Timedelta(days=8)).strftime("%m/%d/%Y %H:%M")
    end = (pd.Timestamp(run_day).normalize() + pd.Timedelta(hours=10)).strftime("%m/%d/%Y %H:%M")
    endpoint = "da_hrl_lmps" if market.lower() == "da" else "rt_hrl_lmps"
    df = fetch_window(endpoint, {"datetime_beginning_ept": f"{start} to {end}"})
    if df.empty: return df
    df = df[df["zone"].astype(str).str.upper() == "BGE"].copy(); df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"], errors="coerce")
    if market.lower() == "da":
        e,c,l,t = "system_energy_price_da","congestion_price_da","marginal_loss_price_da","total_lmp_da"; rename={"energy":"energy_da","congestion":"congestion_da","loss":"loss_da","total":"total_da","n_nodes":"n_nodes_da"}
    else:
        e,c,l,t = "system_energy_price_rt","congestion_price_rt","marginal_loss_price_rt","total_lmp_rt"; rename={"energy":"energy_rt","congestion":"congestion_rt","loss":"loss_rt","total":"total_rt","n_nodes":"n_nodes_rt"}
    return df.groupby("datetime_beginning_ept", as_index=False).agg(energy=(e,"mean"), congestion=(c,"mean"), loss=(l,"mean"), total=(t,"mean"), n_nodes=("pnode_id","nunique")).rename(columns=rename).sort_values("datetime_beginning_ept")

def fetch_actual_load(run_day):
    start = (pd.Timestamp(run_day).normalize() - pd.Timedelta(days=8)).strftime("%m/%d/%Y %H:%M")
    end = (pd.Timestamp(run_day).normalize() + pd.Timedelta(hours=10)).strftime("%m/%d/%Y %H:%M")
    df = fetch_window("hrl_load_prelim", {"datetime_beginning_ept": f"{start} to {end}"})
    if df.empty: return df
    area_col = "load_area" if "load_area" in df.columns else ("area" if "area" in df.columns else "region")
    df = df[df[area_col].astype(str).str.upper() == "MIDATL"].copy(); df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"], errors="coerce")
    return df[["datetime_beginning_ept","prelim_load_avg_hourly"]].rename(columns={"prelim_load_avg_hourly":"load_actual_mw"})

def fetch_forecast_panel_for_live_day(run_day):
    publish_start = (pd.Timestamp(run_day).normalize() - pd.Timedelta(days=1)).strftime("%m/%d/%Y %H:%M")
    publish_end = (pd.Timestamp(run_day).normalize() + pd.Timedelta(hours=10)).strftime("%m/%d/%Y %H:%M")
    df = fetch_window("load_frcstd_hist", {"evaluated_at_ept": f"{publish_start} to {publish_end}", "forecast_area":"MIDATL"})
    if df.empty: return df
    df["evaluated_at_ept"] = pd.to_datetime(df["evaluated_at_ept"], errors="coerce"); df["forecast_hour_beginning_ept"] = pd.to_datetime(df["forecast_hour_beginning_ept"], errors="coerce"); df["forecast_load_mw"] = pd.to_numeric(df["forecast_load_mw"], errors="coerce")
    df = df[df["forecast_area"].astype(str).str.upper() == "MIDATL"].copy()
    runtime = pd.Timestamp(run_day).normalize() + pd.Timedelta(hours=RUNTIME_HOUR)
    target_hours = pd.date_range(pd.Timestamp(run_day).normalize() + pd.Timedelta(days=1), periods=24, freq="h")
    sub = df[(df["evaluated_at_ept"] <= runtime) & (df["forecast_hour_beginning_ept"].isin(target_hours))].copy()
    if sub.empty: return sub
    best_eval = sub.groupby("evaluated_at_ept")["forecast_hour_beginning_ept"].nunique().sort_values(ascending=False).index[0]
    out = sub[sub["evaluated_at_ept"] == best_eval].copy(); out["runtime_hour_ept"] = out["evaluated_at_ept"].dt.floor("h"); out["_run_time"] = runtime
    return out[["runtime_hour_ept","_run_time","evaluated_at_ept","forecast_hour_beginning_ept","forecast_area","forecast_load_mw"]]

def fetch_weather_for_live_day(run_day):
    from meteostat import hourly
    start = (pd.Timestamp(run_day).normalize() - pd.Timedelta(days=8)).to_pydatetime()
    end = (pd.Timestamp(run_day).normalize() + pd.Timedelta(hours=10)).to_pydatetime()
    fetched = hourly("72406", start, end).fetch()
    if fetched is None or len(fetched) == 0:
        return pd.DataFrame()
    wx = fetched.reset_index().rename(columns={"time":"datetime_utc"})
    wx["datetime_utc"] = pd.to_datetime(wx["datetime_utc"], utc=True)
    wx["datetime_beginning_ept"] = wx["datetime_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    keep = ["datetime_beginning_ept","temp","dwpt","rhum","wspd","prcp","pres"]
    wx = wx[[c for c in keep if c in wx.columns]].copy()
    if wx.empty:
        return pd.DataFrame()
    if "temp" in wx.columns:
        base_c = 18.0
        wx["cdh"] = (wx["temp"] - base_c).clip(lower=0)
        wx["hdh"] = (base_c - wx["temp"]).clip(lower=0)
        wx["cdh2"] = wx["cdh"] ** 2
        wx["hdh2"] = wx["hdh"] ** 2
    return wx.sort_values("datetime_beginning_ept").drop_duplicates("datetime_beginning_ept", keep="last")

def merge_live_updates(base_df, parts):
    merged = base_df.copy(); merged[TIME_COL] = pd.to_datetime(merged[TIME_COL])
    for part in parts:
        if part is None or part.empty: continue
        part = part.copy(); part[TIME_COL] = pd.to_datetime(part[TIME_COL])
        merged = merged.merge(part, on=TIME_COL, how="outer", suffixes=("","__new"))
        for col in list(merged.columns):
            if col.endswith("__new"):
                old = col.replace("__new","")
                if old in merged.columns:
                    merged[old] = merged[col].combine_first(merged[old]); merged = merged.drop(columns=[col])
                else:
                    merged = merged.rename(columns={col: old})
    return merged.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last").reset_index(drop=True)

def _friendly_live_error(e: Exception) -> str:
    msg = str(e)
    if "429" in msg:
        return "PJM API rate limit reached. Too many rows or too many requests were sent too quickly, so the app fell back to the saved local snapshot."
    if "Failed to resolve" in msg or "NameResolutionError" in msg:
        return "The app could not reach the PJM API because of a network or DNS connection issue, so it fell back to the saved local snapshot."
    if "401" in msg or "403" in msg:
        return "The PJM API request was rejected. Please check the API key or subscription permissions. The app fell back to the saved local snapshot."
    return f"Fresh fetch failed, so the app used the saved snapshot instead. Details: {msg}"

def build_live_feature_blocks(app_dir, live_day, da_store, rt_store, live_source):
    base_merged = load_curated_merged(app_dir)

    if live_source == "Use latest local live update":
        live_merged = load_live_override_merged(app_dir)
        live_fcst = load_live_override_forecast_panel(app_dir)
        merged = live_merged if isinstance(live_merged, pd.DataFrame) and not live_merged.empty else base_merged

        if isinstance(live_fcst, pd.DataFrame) and not live_fcst.empty:
            fcst = live_fcst
            status = f"Live forecast generated for {pd.Timestamp(live_day).strftime('%Y-%m-%d')} using the latest locally refreshed runtime-safe data."
        else:
            fcst = pd.DataFrame()
            status = (
                f"Live forecast generated for {pd.Timestamp(live_day).strftime('%Y-%m-%d')} using the latest locally refreshed history, "
                f"but no fresh runtime-safe load forecast panel was available, so the load-forecast predictor was omitted for this run."
            )

    elif live_source == "Fetch latest PJM/weather data now":
        try:
            da = ensure_df(fetch_bge_hourly_lmps(live_day, "da"))
            rt = ensure_df(fetch_bge_hourly_lmps(live_day, "rt"))
            load = ensure_df(fetch_actual_load(live_day))
            raw_fcst = ensure_df(fetch_forecast_panel_for_live_day(live_day))
            wx = ensure_df(fetch_weather_for_live_day(live_day))
            fcst = build_runtime_safe_live_forecast_panel(live_day, raw_fcst)
            if fcst.empty:
                fcst = pd.DataFrame()
            merged = merge_live_updates(base_merged, [da, rt, load, wx])
            status = f"Live forecast generated for {pd.Timestamp(live_day).strftime('%Y-%m-%d')} using freshly fetched PJM and weather data with runtime-safe best-publish logic."
        except Exception as e:
            merged = base_merged
            fcst = pd.DataFrame()
            status = _friendly_live_error(e)
    else:
        merged = base_merged
        fcst = load_saved_forecast_panel(app_dir)
        status = f"Live forecast generated for {pd.Timestamp(live_day).strftime('%Y-%m-%d')} using the saved latest snapshot."

    da_live = build_live_block_from_merged(merged, fcst, da_store["feature_cols"], live_day)
    rt_live = build_live_block_from_merged(merged, fcst, rt_store["feature_cols"], live_day)
    return {"da": da_live, "rt": rt_live, "Status Message": status}



def ensure_df(x):
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()
