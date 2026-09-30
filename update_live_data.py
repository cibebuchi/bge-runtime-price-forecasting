
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from requests.exceptions import ConnectionError, Timeout, ChunkedEncodingError

BASE_URL = "https://api.pjm.com/api/v1"
TIMEOUT = 60
ROW_COUNT = 50000
MAX_RETRIES_429 = 10
BACKOFF_BASE = 2.0
MAX_RETRIES_NET = 6
NET_BACKOFF = 5.0
RUNTIME_HOUR = 10

BWI_LAT = 39.1754
BWI_LON = -76.6684
BASE_C = 18.0
TZ = "America/New_York"
TIME_COL = "datetime_beginning_ept"
HORIZON_HOURS = 24
SNAPSHOT_LAGS_H = [1, 2, 3, 6, 12, 24, 48, 72, 168]
SNAPSHOT_ROLL_WINDOWS_H = [6, 24, 72, 168]
FORECAST_COL = "load_forecast_mw"
FORECAST_PRESENT_COL = "load_forecast_present"

def load_pjm_key() -> str:
    env_key = os.environ.get("PJM_API_KEY", "").strip()
    if env_key:
        return env_key
    raise RuntimeError("PJM API key not provided. Paste your PJM key into the sidebar password field to use live refresh.")

def _fmt(dt) -> str:
    return pd.Timestamp(dt).strftime("%m/%d/%Y %H:%M")

def _chunk_ranges_hours(start, end, chunk_hours: int):
    cur = pd.Timestamp(start); end = pd.Timestamp(end)
    while cur <= end:
        nxt = min(cur + pd.Timedelta(hours=chunk_hours) - pd.Timedelta(minutes=1), end)
        yield cur, nxt
        cur = nxt + pd.Timedelta(minutes=1)

def _chunk_ranges_days(start, end, chunk_days: int):
    cur = pd.Timestamp(start); end = pd.Timestamp(end)
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=chunk_days) - pd.Timedelta(minutes=1), end)
        yield cur, nxt
        cur = nxt + pd.Timedelta(minutes=1)

def call_pjm(endpoint: str, params: Dict, attempt_429: int = 0) -> Dict:
    headers = {"Ocp-Apim-Subscription-Key": load_pjm_key()}
    url = f"{BASE_URL}/{endpoint}"
    for net_try in range(MAX_RETRIES_NET + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                if attempt_429 >= MAX_RETRIES_429:
                    raise requests.HTTPError("429 Too Many Requests (max retries hit)", response=r)
                time.sleep(BACKOFF_BASE * (2 ** attempt_429))
                return call_pjm(endpoint, params, attempt_429 + 1)
            if r.status_code >= 400:
                r.raise_for_status()
            return r.json()
        except (ConnectionError, Timeout, ChunkedEncodingError):
            if net_try >= MAX_RETRIES_NET:
                raise
            time.sleep(NET_BACKOFF * (net_try + 1))
    raise RuntimeError("Unreachable")

def fetch_all_rows(endpoint: str, params_base: Dict) -> pd.DataFrame:
    all_items: List[dict] = []
    start_row = 1
    while True:
        params = dict(params_base)
        params["startRow"] = start_row
        params["rowCount"] = ROW_COUNT
        data = call_pjm(endpoint, params)
        items = data.get("items", [])
        total = data.get("totalRows", 0)
        if not items:
            break
        all_items.extend(items)
        start_row += len(items)
        if start_row > total:
            break
        time.sleep(0.15)
    return pd.DataFrame(all_items)

def fetch_bge_hourly_lmps(run_day, market, history_days=8):
    start = pd.Timestamp(run_day).normalize() - pd.Timedelta(days=history_days)
    end = pd.Timestamp(run_day).normalize() + pd.Timedelta(hours=10)
    endpoint = "da_hrl_lmps" if market.lower() == "da" else "rt_hrl_lmps"
    parts = []
    for s, e in _chunk_ranges_hours(start, end, 12):
        df = fetch_all_rows(endpoint, {"datetime_beginning_ept": f"{_fmt(s)} to {_fmt(e)}"})
        if df.empty or "zone" not in df.columns:
            continue
        df = df[df["zone"].astype(str).str.upper() == "BGE"].copy()
        if df.empty:
            continue
        df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"], errors="coerce")
        df["row_is_current"] = df.get("row_is_current", "TRUE").astype(str).str.upper().eq("TRUE")
        if "version_nbr" in df.columns:
            df["version_nbr"] = pd.to_numeric(df["version_nbr"], errors="coerce").fillna(0).astype(int)
        else:
            df["version_nbr"] = 0
        df = df.sort_values(["datetime_beginning_ept", "pnode_id", "row_is_current", "version_nbr"], ascending=[True, True, False, False]).drop_duplicates(subset=["datetime_beginning_ept", "pnode_id"], keep="first")
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.sort_values(["datetime_beginning_ept", "pnode_id", "row_is_current", "version_nbr"], ascending=[True, True, False, False]).drop_duplicates(subset=["datetime_beginning_ept", "pnode_id"], keep="first")
    if market.lower() == "da":
        e, c, l, t = "system_energy_price_da", "congestion_price_da", "marginal_loss_price_da", "total_lmp_da"
        rename = {"energy": "energy_da", "congestion": "congestion_da", "loss": "loss_da", "total": "total_da", "n_nodes": "n_nodes_da"}
    else:
        e, c, l, t = "system_energy_price_rt", "congestion_price_rt", "marginal_loss_price_rt", "total_lmp_rt"
        rename = {"energy": "energy_rt", "congestion": "congestion_rt", "loss": "loss_rt", "total": "total_rt", "n_nodes": "n_nodes_rt"}
    return (
        raw.groupby("datetime_beginning_ept", as_index=False)
        .agg(energy=(e, "mean"), congestion=(c, "mean"), loss=(l, "mean"), total=(t, "mean"), n_nodes=("pnode_id", "nunique"))
        .rename(columns=rename)
        .sort_values("datetime_beginning_ept")
    )

def fetch_actual_load(run_day, history_days=8):
    start = pd.Timestamp(run_day).normalize() - pd.Timedelta(days=history_days)
    end = pd.Timestamp(run_day).normalize() + pd.Timedelta(hours=10)
    parts = []
    for s, e in _chunk_ranges_days(start, end, 7):
        df = fetch_all_rows("hrl_load_prelim", {"datetime_beginning_ept": f"{_fmt(s)} to {_fmt(e)}"})
        if df.empty:
            continue
        area_col = "load_area" if "load_area" in df.columns else ("area" if "area" in df.columns else "region")
        df = df[df[area_col].astype(str).str.upper() == "MIDATL"].copy()
        if df.empty:
            continue
        df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"], errors="coerce")
        parts.append(df[["datetime_beginning_ept", "prelim_load_avg_hourly"]].rename(columns={"prelim_load_avg_hourly": "load_actual_mw"}))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("datetime_beginning_ept").drop_duplicates(subset=["datetime_beginning_ept"], keep="last")

def fetch_next_day_load_forecast_7day_feed(run_day: pd.Timestamp) -> pd.DataFrame:
    target_day = (pd.Timestamp(run_day).normalize() + pd.Timedelta(days=1)).normalize()
    target_start = target_day
    target_end = target_day + pd.Timedelta(hours=23)
    df = fetch_all_rows("load_frcstd_7_day", {"forecast_area": "MID_ATLANTIC_REGION"})
    if df.empty:
        return pd.DataFrame()
    required_cols = ["evaluated_at_datetime_ept", "forecast_datetime_beginning_ept", "forecast_area", "forecast_load_mw"]
    if any(c not in df.columns for c in required_cols):
        return pd.DataFrame()
    df["evaluated_at_datetime_ept"] = pd.to_datetime(df["evaluated_at_datetime_ept"], errors="coerce")
    df["forecast_datetime_beginning_ept"] = pd.to_datetime(df["forecast_datetime_beginning_ept"], errors="coerce")
    df["forecast_load_mw"] = pd.to_numeric(df["forecast_load_mw"], errors="coerce")
    df["datetime_beginning_ept"] = df["forecast_datetime_beginning_ept"]
    next_day = df[(df["datetime_beginning_ept"] >= target_start) & (df["datetime_beginning_ept"] <= target_end)].copy()
    if next_day.empty:
        return pd.DataFrame()
    next_day = next_day.sort_values(["datetime_beginning_ept", "evaluated_at_datetime_ept"]).drop_duplicates(subset=["datetime_beginning_ept"], keep="last").reset_index(drop=True)
    next_day = next_day.rename(columns={"datetime_beginning_ept":"forecast_hour_beginning_ept","evaluated_at_datetime_ept":"evaluated_at_ept"})
    next_day["runtime_hour_ept"] = next_day["evaluated_at_ept"].dt.floor("h")
    next_day["_run_time"] = pd.Timestamp(run_day).normalize() + pd.Timedelta(hours=10)
    next_day["horizon_h"] = ((next_day["forecast_hour_beginning_ept"] - next_day["runtime_hour_ept"]).dt.total_seconds() / 3600.0).round().astype(int)
    out = next_day[["runtime_hour_ept","_run_time","evaluated_at_ept","forecast_hour_beginning_ept","forecast_area","forecast_load_mw","horizon_h"]].copy()
    return out.loc[:, ~out.columns.duplicated()].copy()

def fetch_weather_recent_openmeteo(history_days: int = 7) -> pd.DataFrame:
    now_et = pd.Timestamp.now(tz=TZ)
    end_et = (now_et.floor("h") - pd.Timedelta(hours=1)).tz_localize(None)
    start_et = end_et - pd.Timedelta(days=history_days) + pd.Timedelta(hours=1)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": BWI_LAT,
        "longitude": BWI_LON,
        "start_date": start_et.strftime("%Y-%m-%d"),
        "end_date": end_et.strftime("%Y-%m-%d"),
        "hourly": ",".join(["temperature_2m","relative_humidity_2m","dew_point_2m","precipitation","surface_pressure","wind_speed_10m"]),
        "timezone": TZ,
    }
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "hourly" not in data or "time" not in data["hourly"]:
        return pd.DataFrame()
    hourly = data["hourly"]
    wx = pd.DataFrame({
        "datetime_beginning_ept": pd.to_datetime(hourly["time"]),
        "temp": hourly.get("temperature_2m"),
        "rhum": hourly.get("relative_humidity_2m"),
        "dwpt": hourly.get("dew_point_2m"),
        "prcp": hourly.get("precipitation"),
        "pres": hourly.get("surface_pressure"),
        "wspd": hourly.get("wind_speed_10m"),
    })
    wx = wx[(wx["datetime_beginning_ept"] >= start_et) & (wx["datetime_beginning_ept"] <= end_et)].copy()
    wx = wx.sort_values("datetime_beginning_ept").drop_duplicates(subset=["datetime_beginning_ept"]).reset_index(drop=True)
    if wx.empty or "temp" not in wx.columns:
        return pd.DataFrame()
    for col in ["temp","rhum","dwpt","prcp","pres","wspd"]:
        if col in wx.columns:
            wx[col] = pd.to_numeric(wx[col], errors="coerce")
    wx["cdh"] = (wx["temp"] - BASE_C).clip(lower=0)
    wx["hdh"] = (BASE_C - wx["temp"]).clip(lower=0)
    wx["cdh2"] = wx["cdh"] ** 2
    wx["hdh2"] = wx["hdh"] ** 2
    return wx

def merge_live_updates(base_df, parts):
    merged = base_df.copy()
    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL])
    for part in parts:
        if part is None or not isinstance(part, pd.DataFrame) or part.empty:
            continue
        part = part.copy()
        part[TIME_COL] = pd.to_datetime(part[TIME_COL])
        merged = merged.merge(part, on=TIME_COL, how="outer", suffixes=("", "__new"))
        for col in list(merged.columns):
            if col.endswith("__new"):
                old = col.replace("__new", "")
                if old in merged.columns:
                    merged[old] = merged[col].combine_first(merged[old])
                    merged = merged.drop(columns=[col])
                else:
                    merged = merged.rename(columns={col: old})
    return merged.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last").reset_index(drop=True)

def safe_write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(path)

def safe_write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)

def _json_load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_last_row_leq(df, t, cols):
    sub = df.loc[:t, cols]
    if len(sub) == 0:
        return pd.Series([np.nan] * len(cols), index=cols)
    return sub.iloc[-1]

def add_cyclical_time_features(df, ts_col):
    d = df.copy()
    t = pd.to_datetime(d[ts_col]); hour=t.dt.hour; dow=t.dt.dayofweek; doy=t.dt.dayofyear; month=t.dt.month
    d["is_weekend"]=(dow>=5).astype(int)
    d["hour_sin"]=np.sin(2*np.pi*hour/24.0); d["hour_cos"]=np.cos(2*np.pi*hour/24.0)
    d["dow_sin"]=np.sin(2*np.pi*dow/7.0); d["dow_cos"]=np.cos(2*np.pi*dow/7.0)
    d["doy_sin"]=np.sin(2*np.pi*doy/365.25); d["doy_cos"]=np.cos(2*np.pi*doy/365.25)
    d["month_sin"]=np.sin(2*np.pi*month/12.0); d["month_cos"]=np.cos(2*np.pi*month/12.0)
    return d

def build_snapshot_feature_row(df, runtime, base_cols):
    feat={}; last = df.loc[runtime, base_cols] if runtime in df.index else _safe_last_row_leq(df, runtime, base_cols)
    for c in base_cols:
        feat[f"{c}_at_runtime"] = float(last[c]) if pd.notna(last[c]) else np.nan
    for lag in SNAPSHOT_LAGS_H:
        t_lag = runtime - pd.Timedelta(hours=lag)
        row = df.loc[t_lag, base_cols] if t_lag in df.index else _safe_last_row_leq(df, t_lag, base_cols)
        for c in base_cols:
            feat[f"{c}_lag{lag}h"] = float(row[c]) if pd.notna(row[c]) else np.nan
            a = feat.get(f"{c}_at_runtime", np.nan); b = feat.get(f"{c}_lag{lag}h", np.nan)
            feat[f"{c}_delta{lag}h"] = (a - b) if (np.isfinite(a) and np.isfinite(b)) else np.nan
    for w in SNAPSHOT_ROLL_WINDOWS_H:
        w0 = runtime - pd.Timedelta(hours=w - 1)
        win = df.loc[w0:runtime, base_cols]
        for c in base_cols:
            s = win[c]
            feat[f"{c}_rollmean{w}h"] = float(s.mean()) if s.notna().any() else np.nan
            feat[f"{c}_rollstd{w}h"] = float(s.std()) if s.notna().any() else np.nan
            feat[f"{c}_rollmin{w}h"] = float(s.min()) if s.notna().any() else np.nan
            feat[f"{c}_rollmax{w}h"] = float(s.max()) if s.notna().any() else np.nan
    return feat

def build_proxy_cache_from_merged(app_dir: Path, merged_df: pd.DataFrame, proxy_days: int):
    feat_dir = app_dir / "03_features"
    out_dir = app_dir / "07_proxy_mae"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = merged_df.copy()
    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL])
    merged = merged.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last").set_index(TIME_COL)
    last_obs = pd.Timestamp(merged.index.max())

    # latest run day with full next-day actuals available
    latest_evaluable = None
    for offset in range(0, 5):
        candidate = (last_obs.normalize() - pd.Timedelta(days=offset))
        target_end = candidate.normalize() + pd.Timedelta(days=1, hours=23)
        if target_end <= last_obs:
            latest_evaluable = candidate
            break
    if latest_evaluable is None:
        raise RuntimeError("Not enough refreshed history to build a proxy cache yet.")

    run_days = [latest_evaluable - pd.Timedelta(days=i) for i in range(proxy_days)][::-1]

    base_cols_master = ["total_rt","energy_rt","congestion_rt","loss_rt","total_da","energy_da","congestion_da","loss_da","n_nodes_rt","n_nodes_da","temp","rhum","wspd","prcp","pres","cdh","hdh","cdh2","hdh2","load_actual_mw","gas_price","outage_mw"]
    base_cols = [c for c in base_cols_master if c in merged.columns]

    metadata = {}
    for label, target_col in [("DA_total","total_da"), ("RT_total","total_rt")]:
        feature_cols = _json_load(feat_dir / f"{label}__feature_cols.json")
        rows = []
        for run_day in run_days:
            runtime = pd.Timestamp(run_day).normalize() + pd.Timedelta(hours=RUNTIME_HOUR)
            target_start = pd.Timestamp(run_day).normalize() + pd.Timedelta(days=1)
            target_index = pd.date_range(target_start, periods=24, freq="h")
            if target_index.max() > last_obs:
                continue
            snap = build_snapshot_feature_row(merged, runtime, base_cols)
            X = pd.DataFrame([snap] * len(target_index))
            X["target_timestamp"] = target_index
            X["run_time"] = runtime
            X["run_day"] = pd.Timestamp(run_day).normalize()
            X[FORECAST_COL] = np.nan
            X[FORECAST_PRESENT_COL] = 0
            X = add_cyclical_time_features(X, "target_timestamp")
            for c in feature_cols:
                if c not in X.columns:
                    X[c] = np.nan
            y = merged.reindex(target_index)[target_col].to_numpy(dtype=float) if target_col in merged.columns else np.array([np.nan]*len(target_index))
            X[target_col] = y
            X = X[["run_day","run_time","target_timestamp"] + feature_cols + [target_col]].copy()
            X = X[np.isfinite(X[target_col])].copy()
            if not X.empty:
                rows.append(X)
        if not rows:
            continue
        full = pd.concat(rows, ignore_index=True)
        safe_write_parquet(full, out_dir / f"{label}__ENGINEERED.parquet")
        safe_write_json(feature_cols, out_dir / f"{label}__feature_cols.json")
        metadata[label] = {"rows": int(len(full)), "run_days": [str(pd.Timestamp(d).date()) for d in run_days]}

    safe_write_json({"proxy_days": int(proxy_days), "metadata": metadata}, out_dir / "proxy_cache_status.json")
    return metadata

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--run-day", required=True)
    ap.add_argument("--proxy-days", type=int, default=7)
    ap.add_argument("--refresh-proxy-cache", action="store_true")
    args = ap.parse_args()

    app_dir = Path(args.app_dir)
    run_day = pd.Timestamp(args.run_day).normalize()
    proxy_days = int(args.proxy_days)

    merged_path = app_dir / "02_curated" / "bge_weather_load_merged.parquet"
    live_dir = app_dir / "06_live"
    live_dir.mkdir(parents=True, exist_ok=True)
    if not merged_path.exists():
        raise FileNotFoundError(f"Could not find curated merged file: {merged_path}")

    history_days = max(8, proxy_days + 8 if args.refresh_proxy_cache else 8)

    base_merged = pd.read_parquet(merged_path)
    da = fetch_bge_hourly_lmps(run_day, "da", history_days=history_days)
    rt = fetch_bge_hourly_lmps(run_day, "rt", history_days=history_days)
    load = fetch_actual_load(run_day, history_days=history_days)
    wx = fetch_weather_recent_openmeteo(history_days=history_days)
    fcst = fetch_next_day_load_forecast_7day_feed(run_day)

    merged = merge_live_updates(base_merged, [da, rt, load, wx])

    merged_live_path = live_dir / "bge_weather_load_merged_live.parquet"
    fcst_live_path = live_dir / "load_forecast_runtime10am_next48h_live.parquet"
    meta_path = live_dir / "live_update_status.json"

    safe_write_parquet(merged, merged_live_path)

    forecast_available = isinstance(fcst, pd.DataFrame) and not fcst.empty
    if forecast_available:
        safe_write_parquet(fcst, fcst_live_path)
    elif fcst_live_path.exists():
        fcst_live_path.unlink()

    proxy_meta = None
    if args.refresh_proxy_cache:
        try:
            proxy_meta = build_proxy_cache_from_merged(app_dir, merged, proxy_days=proxy_days)
        except Exception as e:
            proxy_meta = {"error": str(e)}

    status = {
        "status": "ok",
        "run_day": run_day.strftime("%Y-%m-%d"),
        "live_files": {
            "merged_live": str(merged_live_path),
            "forecast_panel_live": str(fcst_live_path) if forecast_available else None,
        },
        "rows": {
            "da": int(len(da)),
            "rt": int(len(rt)),
            "load": int(len(load)),
            "weather": int(len(wx)),
            "forecast_panel": int(len(fcst)) if forecast_available else 0,
            "merged_live": int(len(merged)),
        },
        "forecast_predictor_available": forecast_available,
        "proxy_cache_refreshed": bool(args.refresh_proxy_cache),
        "proxy_window_days": proxy_days,
        "proxy_cache_status": proxy_meta,
        "message": (
            "Live data update completed. Fresh history was refreshed locally. "
            + ("Fresh next-day load forecast panel is available from load_frcstd_7_day."
               if forecast_available else
               "Fresh next-day load forecast panel was not available, so the live app will omit that predictor.")
            + (" Dynamic proxy cache overwritten from the latest live downloads." if args.refresh_proxy_cache else "")
        ),
    }
    safe_write_json(status, meta_path)
    print(status["message"])
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Live data update failed: {type(e).__name__}: {e}", file=sys.stderr)
        raise
