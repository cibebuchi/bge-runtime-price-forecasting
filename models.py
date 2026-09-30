
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from utils import build_live_feature_blocks, build_single_day_train_test, latest_common_run_day

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def make_model(name: str):
    n = name.lower().strip()
    if n == "randomforest":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2))])
    if n == "extratrees":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=420, random_state=42, n_jobs=-1, min_samples_leaf=2))])
    if n == "xgboost":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", XGBRegressor(objective="reg:squarederror", n_estimators=320, learning_rate=0.05, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=4))])
    if n == "voting":
        rf=("rf", RandomForestRegressor(n_estimators=220, random_state=42, n_jobs=-1, min_samples_leaf=2))
        et=("et", ExtraTreesRegressor(n_estimators=260, random_state=42, n_jobs=-1, min_samples_leaf=2))
        xgb=("xgb", XGBRegressor(objective="reg:squarederror", n_estimators=220, learning_rate=0.05, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=4))
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", VotingRegressor([rf, et, xgb]))])
    if n == "stacking":
        estimators=[("rf", RandomForestRegressor(n_estimators=220, random_state=42, n_jobs=-1, min_samples_leaf=2)),
                    ("et", ExtraTreesRegressor(n_estimators=260, random_state=42, n_jobs=-1, min_samples_leaf=2)),
                    ("xgb", XGBRegressor(objective="reg:squarederror", n_estimators=220, learning_rate=0.05, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=4))]
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", StackingRegressor(estimators=estimators, final_estimator=HuberRegressor(), cv=3, passthrough=False))])
    raise ValueError(f"Unsupported model: {name}")

def _single_target_eval(store, eval_day, selected_models, train_window_days):
    X_train, y_train, X_test, y_test, meta = build_single_day_train_test(store["data"], store["target_col"], store["feature_cols"], eval_day, train_window_days)
    rows=[]; preds=[]
    for model_name in selected_models:
        model = make_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append({"Target": store["label"], "Model": model_name, "MAE": mean_absolute_error(y_test, y_pred), "RMSE": rmse(y_test, y_pred)})
        part = meta.copy()
        part["Target"] = store["label"]; part["Model"] = model_name; part["Actual"] = y_test; part["Prediction"] = y_pred
        preds.append(part)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)

def evaluate_models_for_single_day(da_store, rt_store, eval_day, selected_models, train_window_days):
    da_metrics, da_preds = _single_target_eval(da_store, eval_day, selected_models, train_window_days)
    rt_metrics, rt_preds = _single_target_eval(rt_store, eval_day, selected_models, train_window_days)
    da_metrics = da_metrics.rename(columns={"MAE": "DA MAE", "RMSE": "DA RMSE"}).drop(columns=["Target"])
    rt_metrics = rt_metrics.rename(columns={"MAE": "RT MAE", "RMSE": "RT RMSE"}).drop(columns=["Target"])
    summary = da_metrics.merge(rt_metrics, on="Model", how="inner")
    summary = summary.sort_values(["Model"]).reset_index(drop=True)

    spread_details = {}
    for model_name in selected_models:
        da_m = da_preds[da_preds["Model"] == model_name][["Timestamp", "Actual", "Prediction"]].rename(
            columns={"Actual": "DA Actual", "Prediction": "DA Forecast"}
        )
        rt_m = rt_preds[rt_preds["Model"] == model_name][["Timestamp", "Actual", "Prediction"]].rename(
            columns={"Actual": "RT Actual", "Prediction": "RT Forecast"}
        )
        merged = da_m.merge(rt_m, on="Timestamp", how="inner").sort_values("Timestamp")
        merged["Actual Spread (DA - RT)"] = merged["DA Actual"] - merged["RT Actual"]
        merged["Forecast Spread (DA - RT)"] = merged["DA Forecast"] - merged["RT Forecast"]
        merged["Absolute Spread Error"] = (
            merged["Forecast Spread (DA - RT)"] - merged["Actual Spread (DA - RT)"]
        ).abs()
        spread_details[model_name] = merged.reset_index(drop=True)

    preds = pd.concat([da_preds, rt_preds], ignore_index=True)
    return {"summary": summary, "predictions": preds, "spread_details": spread_details}

def live_forecast_for_single_day(app_dir: Path, da_store, rt_store, live_day, selected_models, train_window_days, live_source, proxy_da_store=None, proxy_rt_store=None):
    live_blocks = build_live_feature_blocks(app_dir, live_day, da_store, rt_store, live_source)
    latest_hist_day = latest_common_run_day(da_store["data"], rt_store["data"])
    pred_rows = []
    summary_rows = []

    proxy_source_da = proxy_da_store if proxy_da_store is not None else da_store
    proxy_source_rt = proxy_rt_store if proxy_rt_store is not None else rt_store
    try:
        latest_proxy_day = latest_common_run_day(proxy_source_da["data"], proxy_source_rt["data"])
        proxy = evaluate_models_for_single_day(
            proxy_source_da, proxy_source_rt, latest_proxy_day, selected_models, train_window_days
        )["summary"]
    except Exception:
        proxy = evaluate_models_for_single_day(
            da_store, rt_store, latest_hist_day, selected_models, train_window_days
        )["summary"]

    for store, key in [(da_store, "da"), (rt_store, "rt")]:
        X_train, y_train, _, _, _ = build_single_day_train_test(
            store["data"], store["target_col"], store["feature_cols"], latest_hist_day, train_window_days
        )
        X_live = live_blocks[key][store["feature_cols"]].copy()
        for model_name in selected_models:
            model = make_model(model_name)
            model.fit(X_train, y_train)
            pred = model.predict(X_live)
            pred_rows.append(
                pd.DataFrame({
                    "Target": store["label"],
                    "Model": model_name,
                    "Timestamp": live_blocks[key]["Target Timestamp"],
                    "Prediction": pred,
                })
            )
            proxy_row = proxy.loc[proxy["Model"] == model_name].iloc[0]
            proxy_mae = proxy_row["DA MAE"] if store["label"] == "DA_total" else proxy_row["RT MAE"]
            summary_rows.append({
                "Target": store["label"],
                "Model": model_name,
                "Historical Proxy MAE": proxy_mae,
            })

    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_df["Target"] = pred_df["Target"].replace({"DA_total": "DA Total", "RT_total": "RT Total"})
    summary = pd.DataFrame(summary_rows)
    summary["Target"] = summary["Target"].replace({"DA_total": "DA Total", "RT_total": "RT Total"})
    summary = summary.sort_values(["Target", "Historical Proxy MAE", "Model"]).reset_index(drop=True)

    reference_model = "Stacking" if "Stacking" in selected_models else selected_models[0]
    da_out = pred_df[(pred_df["Target"] == "DA Total") & (pred_df["Model"] == reference_model)][
        ["Timestamp", "Prediction"]
    ].rename(columns={"Prediction": "Forecast DA"})
    rt_out = pred_df[(pred_df["Target"] == "RT Total") & (pred_df["Model"] == reference_model)][
        ["Timestamp", "Prediction"]
    ].rename(columns={"Prediction": "Forecast RT"})
    comparison = da_out.merge(rt_out, on="Timestamp", how="inner")
    comparison["Forecast Spread (DA - RT)"] = comparison["Forecast DA"] - comparison["Forecast RT"]

    return {
        "predictions": pred_df,
        "summary": summary,
        "comparison": comparison,
        "Reference Model": reference_model,
        "Status Message": live_blocks["Status Message"],
    }
