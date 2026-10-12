#!/usr/bin/env python3
"""Leakage-safe temporal analysis and publication artifact generator for SBGR."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import lightgbm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from make_figures import draw_all

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parent
DATA_DIR = Path(os.environ.get("SBGR_DATA_DIR", REPO / "data"))
OUTPUT_DIR = Path(os.environ.get("SBGR_OUTPUT_DIR", REPO / "output"))
ARCHIVED_RESULTS = REPO / "results" / "analysis_results.json"

SEED = 20260720
HORIZONS = (15, 30, 60)
RECURSIVE_STEPS = (15, 30, 45, 60)
RUNWAY_HEADING_10 = 73.69
RUNWAY_HEADING_28 = 253.67
POSITIVE_LABEL = 1  # minority configuration 28L|28R
MODEL_LABELS = OrderedDict([
    ("majority", "Majority"),
    ("persistence", "Persistence"),
    ("climatology", "Climatology"),
    ("wind_rule", "Wind rule"),
    ("logistic", "Logistic regression"),
    ("random_forest", "Random forest"),
    ("hist_gradient_boosting", "Gradient boosting"),
])
EXTRA_MODEL_LABELS = OrderedDict([
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),
    ("mlp", "Neural network"),
    ("discrete_choice", "Discrete choice"),
])
ALL_MODEL_LABELS = OrderedDict(list(MODEL_LABELS.items()) + list(EXTRA_MODEL_LABELS.items()))

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def build_interval_data(data_dir: Path) -> pd.DataFrame:
    script = CODE_DIR / "prepare_data.R"
    cached = os.environ.get("SBGR_GRID_CSV")
    if cached and Path(cached).exists():
        df = pd.read_csv(cached)
    else:
        with tempfile.TemporaryDirectory(prefix="jatm_sbgr_") as tmp:
            csv_path = Path(tmp) / "analysis_grid.csv"
            subprocess.run(["Rscript", str(script), str(data_dir), str(csv_path)], check=True)
            df = pd.read_csv(csv_path)
    for col in ("time", "metar_time"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def add_issue_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    state = d["state_configuration"]
    state_change = state.ne(state.shift()).astype(int)
    run_id = state_change.cumsum()
    run_slots = d.groupby(run_id, sort=False).cumcount() + 1

    d["current_configuration"] = state.shift(1)
    d["current_is_28"] = (d["current_configuration"] == "28L|28R").astype(float)
    d.loc[d["current_configuration"].isna(), "current_is_28"] = np.nan
    d["time_since_change_hours"] = run_slots.shift(1) * 0.25
    d["recent_transition_count_2h"] = state_change.shift(1).rolling(8, min_periods=1).sum()
    # Individual change indicators leaving the two-hour window as a recursive forecast
    # advances one step at a time. They are observed history at issue time and are used
    # only by the recursive comparator, never as model predictors.
    for lag in (6, 7, 8):
        d[f"state_change_lag_{lag}"] = state_change.shift(lag)
    d["recent_arrivals_15m"] = d["arr_operations"].shift(1)
    d["recent_departures_15m"] = d["dep_operations"].shift(1)
    d["recent_arrivals_60m"] = d["arr_operations"].shift(1).rolling(4, min_periods=1).sum()
    d["recent_departures_60m"] = d["dep_operations"].shift(1).rolling(4, min_periods=1).sum()

    radians = np.deg2rad(d["wind_direction"].astype(float))
    d["wind_sin"] = np.sin(radians)
    d["wind_cos"] = np.cos(radians)
    # METAR reports wind direction relative to true north, so the runway-relative
    # components are projected onto the published true bearings of 73.69 and
    # 253.67 degrees. The magnetic bearings 95/275 degrees name the runways but
    # are not a valid reference frame for these components.
    delta10 = np.deg2rad(d["wind_direction"].astype(float) - RUNWAY_HEADING_10)
    delta28 = np.deg2rad(d["wind_direction"].astype(float) - RUNWAY_HEADING_28)
    d["headwind_10"] = d["wind_speed"] * np.cos(delta10)
    d["headwind_28"] = d["wind_speed"] * np.cos(delta28)
    d["crosswind_10"] = d["wind_speed"] * np.sin(delta10)
    d["tailwind_10"] = np.maximum(0.0, -d["headwind_10"])
    d["tailwind_28"] = np.maximum(0.0, -d["headwind_28"])
    d["ceiling_capped"] = d["ceiling"].clip(upper=10_000)
    return d


FEATURE_GROUPS = OrderedDict([
    ("current", ["current_is_28", "time_since_change_hours", "recent_transition_count_2h"]),
    ("wind", ["wind_speed", "gust_speed", "wind_sin", "wind_cos", "headwind_10",
              "headwind_28", "crosswind_10", "tailwind_10", "tailwind_28"]),
    ("weather", ["visibility", "ceiling_capped", "temperature", "dew_point",
                 "relative_humidity", "pressure", "ifr", "lifr", "ts"]),
    ("calendar", ["target_tod_sin", "target_tod_cos", "target_doy_sin",
                  "target_doy_cos", "target_dow_sin", "target_dow_cos"]),
    ("planned", ["target_scheduled_arrivals", "target_scheduled_departures"]),
    ("recent_ops", ["recent_arrivals_15m", "recent_departures_15m",
                    "recent_arrivals_60m", "recent_departures_60m"]),
])
ALL_FEATURES = sum(FEATURE_GROUPS.values(), [])

GROUP_LABELS = OrderedDict([
    ("current", "Current operational state"),
    ("wind", "Wind"),
    ("weather", "Other weather"),
    ("calendar", "Calendar"),
    ("planned", "Planned demand"),
    ("recent_ops", "Recent operations"),
])

# Printed definitions for the final feature set. The availability column states the
# latest information that may enter each predictor at issue time t.
FEATURE_DEFINITIONS = OrderedDict([
    ("current_is_28", ("Configuration indicator", "1 if the last completed interval was 28L$|$28R, 0 otherwise", "interval ending at $t$")),
    ("time_since_change_hours", ("State duration", "hours since the last configuration change, in 15-min steps", "interval ending at $t$")),
    ("recent_transition_count_2h", ("Recent instability", "configuration changes in the preceding eight completed intervals", "interval ending at $t$")),
    ("wind_speed", ("Wind speed", "reported sustained wind speed (kt)", "latest METAR at or before $t$")),
    ("gust_speed", ("Gust speed", "reported gust speed (kt); zero when not reported", "latest METAR at or before $t$")),
    ("wind_sin", ("Wind direction sine", "$\\sin$ of reported wind direction", "latest METAR at or before $t$")),
    ("wind_cos", ("Wind direction cosine", "$\\cos$ of reported wind direction", "latest METAR at or before $t$")),
    ("headwind_10", ("Headwind, direction 10", "signed component along true bearing 73.7$^\\circ$ (kt)", "latest METAR at or before $t$")),
    ("headwind_28", ("Headwind, direction 28", "signed component along true bearing 253.7$^\\circ$ (kt)", "latest METAR at or before $t$")),
    ("crosswind_10", ("Crosswind", "signed component normal to the runway axis (kt)", "latest METAR at or before $t$")),
    ("tailwind_10", ("Tailwind, direction 10", "$\\max(0,-\\text{headwind}_{10})$ (kt)", "latest METAR at or before $t$")),
    ("tailwind_28", ("Tailwind, direction 28", "$\\max(0,-\\text{headwind}_{28})$ (kt)", "latest METAR at or before $t$")),
    ("visibility", ("Visibility", "prevailing visibility (NM)", "latest METAR at or before $t$")),
    ("ceiling_capped", ("Ceiling", "lowest broken or overcast layer (ft), capped at 10,000, which also absorbs the no-ceiling code", "latest METAR at or before $t$")),
    ("temperature", ("Temperature", "air temperature ($^\\circ$C)", "latest METAR at or before $t$")),
    ("dew_point", ("Dew point", "dew-point temperature ($^\\circ$C)", "latest METAR at or before $t$")),
    ("relative_humidity", ("Relative humidity", "derived from temperature and dew point (\\%)", "latest METAR at or before $t$")),
    ("pressure", ("Pressure", "reported altimeter setting (inHg)", "latest METAR at or before $t$")),
    ("ifr", ("IFR indicator", "1 if ceiling $<$ 1,000 ft or visibility $<$ 3 NM", "latest METAR at or before $t$")),
    ("lifr", ("Low-IFR indicator", "1 if ceiling $<$ 500 ft or visibility $<$ 1 NM", "latest METAR at or before $t$")),
    ("ts", ("Thunderstorm indicator", "1 if the report contains a thunderstorm group", "latest METAR at or before $t$")),
    ("target_tod_sin", ("Time of day, sine", "$\\sin(2\\pi\\,\\text{hour}/24)$ at the target time", "deterministic")),
    ("target_tod_cos", ("Time of day, cosine", "$\\cos(2\\pi\\,\\text{hour}/24)$ at the target time", "deterministic")),
    ("target_doy_sin", ("Day of year, sine", "$\\sin(2\\pi\\,\\text{day}/365.25)$ at the target time", "deterministic")),
    ("target_doy_cos", ("Day of year, cosine", "$\\cos(2\\pi\\,\\text{day}/365.25)$ at the target time", "deterministic")),
    ("target_dow_sin", ("Day of week, sine", "$\\sin(2\\pi\\,\\text{weekday}/7)$ at the target time", "deterministic")),
    ("target_dow_cos", ("Day of week, cosine", "$\\cos(2\\pi\\,\\text{weekday}/7)$ at the target time", "deterministic")),
    ("target_scheduled_arrivals", ("Scheduled arrivals", "arrivals registered for the target 15-min bin", "published schedule")),
    ("target_scheduled_departures", ("Scheduled departures", "departures registered for the target 15-min bin", "published schedule")),
    ("recent_arrivals_15m", ("Recent arrivals", "landings in the last completed 15-min interval", "interval ending at $t$")),
    ("recent_departures_15m", ("Recent departures", "take-offs in the last completed 15-min interval", "interval ending at $t$")),
    ("recent_arrivals_60m", ("Rolling arrivals", "landings in the four last completed intervals", "interval ending at $t$")),
    ("recent_departures_60m", ("Rolling departures", "take-offs in the four last completed intervals", "interval ending at $t$")),
])

# Restricted utility specification used for the discrete-choice comparator, following
# the variable classes of the published airport-configuration choice model.
DISCRETE_CHOICE_FEATURES = [
    "current_is_28", "headwind_10", "headwind_28", "crosswind_10",
    "tailwind_10", "tailwind_28", "wind_speed", "visibility", "ceiling_capped",
    "target_scheduled_arrivals", "target_scheduled_departures",
    "target_tod_sin", "target_tod_cos",
]


def horizon_frame(base: pd.DataFrame, horizon: int) -> pd.DataFrame:
    d = base.copy()
    d["issue_time"] = d["time"]
    d["target_time"] = d["issue_time"] + pd.Timedelta(minutes=horizon)
    target = base[["time", "primary_configuration", "primary_label_source",
                   "scheduled_arrivals", "scheduled_departures"]].rename(columns={
        "time": "target_time",
        "primary_configuration": "target_configuration",
        "primary_label_source": "target_label_source",
        "scheduled_arrivals": "target_scheduled_arrivals",
        "scheduled_departures": "target_scheduled_departures",
    })
    d = d.merge(target, on="target_time", how="inner", validate="one_to_one")
    local = d["target_time"].dt.tz_convert("America/Sao_Paulo")
    tod = local.dt.hour + local.dt.minute / 60.0
    doy = local.dt.dayofyear
    dow = local.dt.dayofweek
    d["target_tod_sin"] = np.sin(2 * np.pi * tod / 24)
    d["target_tod_cos"] = np.cos(2 * np.pi * tod / 24)
    d["target_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["target_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["target_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    d["target_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    d["target_month"] = local.dt.month
    d["target_hour_block"] = (local.dt.hour // 4).astype(int)
    # Operational-day resampling and alert-rate denominators use São Paulo local dates.
    d["target_date"] = local.dt.date
    d["y"] = (d["target_configuration"] == "28L|28R").astype(int)
    valid = (d["target_configuration"].notna() & d["current_configuration"].notna()
             & d["metar_age_minutes"].between(0, 120, inclusive="both"))
    return d.loc[valid].reset_index(drop=True)


TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
TEST_END = pd.Timestamp("2024-06-30 23:45", tz="UTC")


def dev_test_split(frame: pd.DataFrame):
    """Development and locked-test partitions under a strictly temporal boundary.

    A development row is admissible only if its target time precedes the earliest
    issue time of the test partition. Without that condition a model applied to the
    first test forecasts would have been fitted on outcomes that had not yet
    occurred when those forecasts were issued, because the target of a development
    row at the boundary can fall after the issue time of a test row at a longer
    horizon. The condition removes the affected boundary rows from development.
    """
    test = frame[frame["target_time"].between(TEST_START, TEST_END)]
    dev = frame[frame["target_time"] < TEST_START]
    if len(test):
        dev = dev[dev["target_time"] < test["issue_time"].min()]
    return dev.reset_index(drop=True), test.reset_index(drop=True)


def development_split(frame: pd.DataFrame) -> pd.DataFrame:
    return dev_test_split(frame)[0]


def rolling_folds(d: pd.DataFrame):
    bounds = [
        ("2023Q1", "2023-01-01", "2023-04-01"),
        ("2023Q2", "2023-04-01", "2023-07-01"),
        ("2023Q3", "2023-07-01", "2023-10-01"),
        ("2023Q4", "2023-10-01", "2024-01-01"),
    ]
    for name, start, end in bounds:
        start_t, end_t = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        va = np.flatnonzero(d["target_time"].ge(start_t).to_numpy()
                            & d["target_time"].lt(end_t).to_numpy())
        if not len(va):
            continue
        # The same boundary condition as the development/test split: a training row
        # is admissible only if its target had already been observed when the
        # earliest validation forecast of the fold was issued.
        cutoff = d["issue_time"].to_numpy()[va].min()
        tr = np.flatnonzero(d["target_time"].lt(start_t).to_numpy()
                            & (d["target_time"].to_numpy() < cutoff))
        if len(tr) and len(va):
            yield name, tr, va


MODEL_GRIDS = {
    "logistic": [
        {"C": 0.1, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
    ],
    "random_forest": [
        {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 5,
         "max_features": "sqrt", "class_weight": "balanced_subsample"},
    ],
    "hist_gradient_boosting": [
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    ],
}


def estimator(name: str, params: dict, balanced: bool = True):
    """Return the pipeline for one model family.

    ``balanced`` controls whether the family applies inverse-prevalence weighting.
    It is disabled when the imbalance experiment resamples the training folds, so
    that reweighting and resampling are never applied simultaneously.
    """
    if name == "logistic":
        settings = dict(params)
        if not balanced:
            settings["class_weight"] = None
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs",
                                         random_state=SEED, **settings)),
        ])
    if name == "discrete_choice":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs",
                                         class_weight=None, random_state=SEED)),
        ])
    if name == "random_forest":
        settings = dict(params)
        if not balanced:
            settings["class_weight"] = None
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=SEED, n_jobs=-1, **settings)),
        ])
    if name == "hist_gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=180, random_state=SEED,
                class_weight="balanced" if balanced else None, **params)),
        ])
    if name == "xgboost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=180, learning_rate=0.05, max_leaves=15, max_depth=0,
                grow_policy="lossguide", reg_lambda=1.0, tree_method="hist",
                random_state=SEED, n_jobs=-1, eval_metric="logloss", **params)),
        ])
    if name == "lightgbm":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(
                n_estimators=180, learning_rate=0.05, num_leaves=15, max_depth=-1,
                reg_lambda=1.0, class_weight="balanced" if balanced else None,
                random_state=SEED, n_jobs=-1, verbose=-1, **params)),
        ])
    if name == "mlp":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                                    alpha=1e-3, learning_rate_init=1e-3, max_iter=400,
                                    early_stopping=True, n_iter_no_change=10,
                                    random_state=SEED, **params)),
        ])
    raise KeyError(name)


def resample_training(x: pd.DataFrame, y: pd.Series, strategy: str):
    """Resample one training fold. Resampling never touches validation or test rows."""
    if strategy == "none":
        return x, y
    if strategy == "oversample":
        sampler = RandomOverSampler(random_state=SEED)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=SEED)
    elif strategy == "smote":
        sampler = SMOTE(random_state=SEED, k_neighbors=5)
    else:
        raise KeyError(strategy)
    filled = x.fillna(x.median(numeric_only=True))
    xr, yr = sampler.fit_resample(filled, y)
    return xr, yr


EXTRA_GRIDS = {
    "xgboost": [{}],
    "lightgbm": [{}],
    "mlp": [{}],
    "discrete_choice": [{}],
}


def select_hyperparameters(dev: pd.DataFrame, features: list[str], model_name: str,
                           grid=None):
    records = []
    for params in (grid if grid is not None else MODEL_GRIDS[model_name]):
        fold_scores = []
        for fold, tr, va in rolling_folds(dev):
            model = estimator(model_name, params)
            fit_pipeline(model, model_name, dev.iloc[tr][features], dev.iloc[tr]["y"])
            pred = model.predict(dev.iloc[va][features])
            fold_scores.append({
                "fold": fold,
                "balanced_accuracy": balanced_accuracy_score(dev.iloc[va]["y"], pred),
                "macro_f1": f1_score(dev.iloc[va]["y"], pred, average="macro"),
            })
        records.append({
            "params": params,
            "mean_balanced_accuracy": float(np.mean([x["balanced_accuracy"] for x in fold_scores])),
            "mean_macro_f1": float(np.mean([x["macro_f1"] for x in fold_scores])),
            "fold_scores": fold_scores,
        })
    best = max(records, key=lambda x: (x["mean_balanced_accuracy"], x["mean_macro_f1"]))
    return best, records


def logit_calibrator(raw: np.ndarray, y: np.ndarray):
    z = np.log(np.clip(raw, 1e-5, 1 - 1e-5) / np.clip(1 - raw, 1e-5, 1 - 1e-5))
    cal = LogisticRegression(C=1e4, solver="lbfgs", random_state=SEED)
    cal.fit(z.reshape(-1, 1), y)
    return cal


def apply_calibrator(cal, raw: np.ndarray) -> np.ndarray:
    z = np.log(np.clip(raw, 1e-5, 1 - 1e-5) / np.clip(1 - raw, 1e-5, 1 - 1e-5))
    return cal.predict_proba(z.reshape(-1, 1))[:, 1]


def fit_pipeline(pipeline, name, x, y, balanced=True):
    """Fit one pipeline, supplying inverse-prevalence sample weights where the
    underlying implementation exposes no class-weight argument."""
    if name == "xgboost" and balanced:
        weights = compute_sample_weight("balanced", y)
        pipeline.fit(x, y, model__sample_weight=weights)
    else:
        pipeline.fit(x, y)
    return pipeline


def choose_threshold(oof: pd.DataFrame, prob: np.ndarray, criterion: str,
                     grid: np.ndarray, alert_budget: float = 1.0) -> float:
    """Select an operating threshold from rolling out-of-fold predictions only."""
    if criterion == "balanced_accuracy":
        return float(max(grid, key=lambda t: balanced_accuracy_score(oof["y"], prob >= t)))
    current = oof["current"].to_numpy()
    actual = oof["y"].to_numpy() != current
    days = max(1, oof["date"].nunique())
    best, best_key = float(grid[0]), None
    for t in grid:
        alert = (prob >= t).astype(int) != current
        tp = float(np.sum(actual & alert))
        fp = float(np.sum(~actual & alert))
        fn = float(np.sum(actual & ~alert))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if criterion == "transition_f1":
            key = (f1, recall)
        elif criterion == "alert_budget":
            key = (recall if fp / days <= alert_budget else -1.0, f1)
        else:
            raise KeyError(criterion)
        if best_key is None or key > best_key:
            best, best_key = float(t), key
    return best


def fitted_model_with_calibration(dev, test, features, name, params, *,
                                  balanced=True, resampling="none",
                                  threshold_criterion="balanced_accuracy",
                                  threshold_grid=None, alert_budget=1.0):
    oof_parts = []
    for fold, tr, va in rolling_folds(dev):
        model = estimator(name, params, balanced=balanced)
        x_tr, y_tr = resample_training(dev.iloc[tr][features], dev.iloc[tr]["y"], resampling)
        fit_pipeline(model, name, x_tr, y_tr, balanced=balanced)
        raw = model.predict_proba(dev.iloc[va][features])[:, 1]
        oof_parts.append(pd.DataFrame({
            "raw": raw,
            "y": dev.iloc[va]["y"].to_numpy(),
            "current": dev.iloc[va]["current_is_28"].astype(int).to_numpy(),
            "date": dev.iloc[va]["target_date"].to_numpy(),
        }))
    oof = pd.concat(oof_parts, ignore_index=True)
    cal = logit_calibrator(oof["raw"].to_numpy(), oof["y"].to_numpy())
    oof_prob = apply_calibrator(cal, oof["raw"].to_numpy())
    grid = np.linspace(0.20, 0.80, 121) if threshold_grid is None else threshold_grid
    threshold = choose_threshold(oof, oof_prob, threshold_criterion, grid, alert_budget)
    final = estimator(name, params, balanced=balanced)
    x_dev, y_dev = resample_training(dev[features], dev["y"], resampling)
    fit_pipeline(final, name, x_dev, y_dev, balanced=balanced)
    test_prob = apply_calibrator(cal, final.predict_proba(test[features])[:, 1])
    oof["prob"] = oof_prob
    return final, cal, float(threshold), test_prob, oof


def fit_climatology(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    global_p = train["y"].mean()
    grouped = train.groupby(["target_month", "target_hour_block"])["y"].agg(["sum", "count"])
    smooth = (grouped["sum"] + 20 * global_p) / (grouped["count"] + 20)
    keys = pd.MultiIndex.from_frame(test[["target_month", "target_hour_block"]])
    return smooth.reindex(keys).fillna(global_p).to_numpy()


def wind_prediction(d: pd.DataFrame, margin: float) -> np.ndarray:
    current = d["current_is_28"].astype(int).to_numpy().copy()
    advantage = (d["headwind_28"] - d["headwind_10"]).to_numpy()
    current[advantage > margin] = 1
    current[advantage < -margin] = 0
    return current


def expected_calibration_error(y, prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(prob, edges, right=True) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = ids == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - prob[mask].mean())
    return float(ece)


def metrics(y, pred, prob):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "minority_precision": float(precision_score(y, pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "minority_recall": float(recall_score(y, pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "minority_f1": float(f1_score(y, pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "brier": float(brier_score_loss(y, prob)),
        "ece": expected_calibration_error(np.asarray(y), np.asarray(prob)),
    }


def block_bootstrap(d, pred, prob, n_boot=500):
    # The environment override exists only for pipeline smoke tests; reported
    # results always use the default 500 paired operational-day resamples.
    n_boot = int(os.environ.get("SBGR_NBOOT", n_boot))
    rng = np.random.default_rng(SEED)
    date_values = np.array(sorted(pd.unique(d["target_date"])), dtype=object)
    index_by_day = {day: np.flatnonzero(d["target_date"].to_numpy() == day) for day in date_values}
    y = d["y"].to_numpy()
    out = {k: [] for k in ("accuracy", "balanced_accuracy", "macro_f1", "minority_f1", "mcc", "brier")}
    sampled_indices = []
    for _ in range(n_boot):
        sample_days = rng.choice(date_values, size=len(date_values), replace=True)
        idx = np.concatenate([index_by_day[x] for x in sample_days])
        sampled_indices.append(idx)
        m = metrics(y[idx], pred[idx], prob[idx])
        for key in out:
            out[key].append(m[key])
    cis = {key: [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
           for key, vals in out.items()}
    return cis, sampled_indices


def transition_metrics(d, pred):
    y = d["y"].to_numpy()
    current = d["current_is_28"].astype(int).to_numpy()
    actual = y != current
    alert = pred != current
    tp = int(np.sum(actual & alert))
    fp = int(np.sum(~actual & alert))
    fn = int(np.sum(actual & ~alert))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    days = max(1, d["target_date"].nunique())
    return {
        "transition_precision": precision,
        "transition_recall": recall,
        "transition_f1": f1,
        "missed_transition_intervals": fn,
        "false_alerts_per_day": fp / days,
        "transition_intervals": int(actual.sum()),
    }


def event_warning_metrics(events, candidate_sets, tolerance_minutes=15):
    """Create deterministic one-to-one event/warning pairs.

    Each transition-alert forecast is first assigned to at most one directionally
    matching event: the event with the smallest absolute target-time error inside
    the tolerance window (ties go to the earlier event). For each event, the
    earliest issued assigned forecast is retained; ties prefer smaller absolute
    target-time error and then the shorter horizon. Other forecasts for that event
    are duplicates and do not affect event coverage or summary timing statistics.
    """
    event_records = events.reset_index(drop=True).copy()
    event_records["event_id"] = np.arange(len(event_records))
    event_records["new_y"] = (
        event_records["primary_configuration"] == "28L|28R"
    ).astype(int)
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    assignments = []
    transition_alerts = 0

    for horizon, tf, pred in candidate_sets:
        alert_rows = np.flatnonzero(
            pred != tf["current_is_28"].astype(int).to_numpy()
        )
        transition_alerts += len(alert_rows)
        for j in alert_rows:
            row = tf.iloc[j]
            eligible = event_records[
                event_records["new_y"].eq(int(pred[j]))
                & event_records["time"].gt(row["issue_time"])
                & event_records["time"].between(
                    row["target_time"] - tolerance,
                    row["target_time"] + tolerance,
                )
            ].copy()
            if eligible.empty:
                continue
            eligible["abs_target_error"] = (
                eligible["time"] - row["target_time"]
            ).abs()
            matched_event = eligible.sort_values(
                ["abs_target_error", "time", "event_id"]
            ).iloc[0]
            timing = (
                row["target_time"] - matched_event["time"]
            ).total_seconds() / 60
            assignments.append({
                "event_id": int(matched_event["event_id"]),
                "event_time": matched_event["time"],
                "issue_time": row["issue_time"],
                "target_time": row["target_time"],
                "horizon": int(horizon),
                "lead": (
                    matched_event["time"] - row["issue_time"]
                ).total_seconds() / 60,
                "timing": timing,
                "abs_timing": abs(timing),
            })

    selected = []
    duplicate_events = 0
    if assignments:
        assigned = pd.DataFrame(assignments)
        for _, group in assigned.groupby("event_id", sort=True):
            if len(group) > 1:
                duplicate_events += 1
            selected.append(group.sort_values(
                ["issue_time", "abs_timing", "horizon", "target_time"]
            ).iloc[0])

    warned_leads = [float(x["lead"]) for x in selected]
    timing_errors = [float(x["timing"]) for x in selected]
    warned = len(selected)
    n_events = len(event_records)
    return {
        "events": int(n_events),
        "warned_within_tolerance": int(warned),
        "event_recall": float(warned / n_events) if n_events else 0.0,
        "median_warning_lead_minutes": float(np.median(warned_leads)) if warned_leads else 0.0,
        "median_absolute_timing_error_minutes": (
            float(np.median(np.abs(timing_errors))) if timing_errors else 0.0
        ),
        "correct_direction_share": 1.0 if warned else 0.0,
        "missed_events": int(n_events - warned),
        "tolerance_minutes": int(tolerance_minutes),
        "transition_alert_forecasts": int(transition_alerts),
        "eligible_forecasts_assigned": int(len(assignments)),
        "events_with_duplicate_forecasts": int(duplicate_events),
        "matching_rule": (
            "Each transition-alert forecast is assigned to the directionally matching "
            "event with closest target time within tolerance, with earlier-event ties; "
            "the earliest issued forecast per event is retained, with absolute target-"
            "time error and shorter horizon as tie-breakers."
        ),
    }


def recursive_multistep(base: pd.DataFrame, features: list[str], name: str, params: dict):
    """Recursive multi-step configuration forecast.

    A single one-step (15-minute) classifier is applied repeatedly, feeding its own
    predicted state into the state-dependent predictors of a later step, as in
    recursive multi-step configuration prediction. Calendar and schedule predictors
    advance with the step because they are deterministic or published in advance;
    weather and realized-movement predictors remain frozen at their issue-time values
    because no forecast weather is available in the SBGR data.

    The lag between the state predictor and the target is two intervals, so the state
    consumed at each step is set out in the loop below rather than taken from the
    immediately preceding prediction.
    """
    frames = {k: horizon_frame(base, k) for k in RECURSIVE_STEPS}
    common = None
    for k in RECURSIVE_STEPS:
        times = set(frames[k]["issue_time"])
        common = times if common is None else (common & times)
    aligned = {}
    for k in RECURSIVE_STEPS:
        f = frames[k]
        aligned[k] = f[f["issue_time"].isin(common)].sort_values("issue_time").reset_index(drop=True)

    base15 = frames[15]
    test_mask = aligned[15]["target_time"].between(TEST_START, TEST_END)
    step_test = {k: aligned[k].loc[test_mask.to_numpy()].reset_index(drop=True) for k in RECURSIVE_STEPS}
    dev = base15[base15["target_time"] < TEST_START]
    if len(step_test[15]):
        dev = dev[dev["target_time"] < step_test[15]["issue_time"].min()]
    dev = dev.reset_index(drop=True)

    _, _, threshold, _, _ = fitted_model_with_calibration(
        dev, step_test[15], features, name, params)
    model = estimator(name, params)
    fit_pipeline(model, name, dev[features], dev["y"])
    oof_parts = []
    for fold, tr, va in rolling_folds(dev):
        fold_model = estimator(name, params)
        fit_pipeline(fold_model, name, dev.iloc[tr][features], dev.iloc[tr]["y"])
        oof_parts.append(pd.DataFrame({
            "raw": fold_model.predict_proba(dev.iloc[va][features])[:, 1],
            "y": dev.iloc[va]["y"].to_numpy()}))
    oof = pd.concat(oof_parts, ignore_index=True)
    cal = logit_calibrator(oof["raw"].to_numpy(), oof["y"].to_numpy())

    # Step j has pseudo-issue time t+15j and target interval t+15(j+1). Because the
    # configuration predictor is the last *completed* interval, the state a step
    # consumes is the interval at t+15(j-1), two steps before its own target, not the
    # interval it has just predicted. The feature sequence is therefore
    #   j=0: interval t-15, observed at the real issue time;
    #   j=1: interval t, still in progress at the real issue time and carried forward;
    #   j>=2: the prediction issued by step j-2, which is the recursion itself.
    # Substituting the immediately preceding prediction instead would advance the
    # state by one interval relative to the convention under which the one-step model
    # was fitted.
    observed = step_test[15]["current_is_28"].astype(int).to_numpy()
    feature_state = observed.copy()
    duration = step_test[15]["time_since_change_hours"].astype(float).to_numpy().copy()
    instability = step_test[15]["recent_transition_count_2h"].astype(float).to_numpy().copy()
    # Recent instability is a rolling count over the eight completed intervals before the
    # pseudo-issue time, so advancing one step adds the change entering the window and
    # removes the one leaving it. The entering change is the recursion's own; the leaving
    # change is observed history, at lags 8, 7 and 6 for the three advances taken here.
    leaving = [step_test[15][f"state_change_lag_{lag}"].fillna(0.0).to_numpy().astype(float)
               for lag in (8, 7, 6)]
    predictions, issued = {}, []
    for j, k in enumerate(RECURSIVE_STEPS):
        x = step_test[k][features].copy()
        x["current_is_28"] = feature_state.astype(float)
        x["time_since_change_hours"] = duration
        x["recent_transition_count_2h"] = instability
        prob = apply_calibrator(cal, model.predict_proba(x)[:, 1])
        pred = (prob >= threshold).astype(int)
        predictions[k] = {"pred": pred, "prob": prob}
        issued.append(pred)
        next_state = observed if j == 0 else issued[j - 1]
        changed = (next_state != feature_state).astype(float)
        duration = np.where(changed > 0, 0.25, duration + 0.25)
        if j < len(leaving):
            instability = np.clip(instability + changed - leaving[j], 0.0, 8.0)
        feature_state = next_state
    return step_test, predictions, float(threshold)


def summarize_predictions(frame: pd.DataFrame, pred: np.ndarray, prob: np.ndarray,
                          events=None, horizon=None, n_boot=0):
    point = metrics(frame["y"], pred, prob)
    point.update(transition_metrics(frame, pred))
    if n_boot:
        point["ci"], _ = block_bootstrap(frame, pred, prob, n_boot=n_boot)
    if events is not None and horizon is not None:
        em = event_warning_metrics(events, [(horizon, frame, pred)], tolerance_minutes=15)
        point["event_recall"] = em["event_recall"]
        point["warned_events"] = em["warned_within_tolerance"]
        point["median_warning_lead_minutes"] = em["median_warning_lead_minutes"]
    return point


def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")




def stacked_header(text: str) -> str:
    """Render a column heading on up to two balanced lines.

    Single-line headings set the natural width of narrow numeric columns, which is
    what forced the wide tables to be scaled down. Stacking them lets the data
    determine the width instead.
    """
    words = str(text).split()
    if len(words) < 2:
        return str(text)
    target = len(str(text)) / 2
    best, cut = None, 1
    for i in range(1, len(words)):
        top = len(" ".join(words[:i]))
        score = abs(top - target)
        if best is None or score < best:
            best, cut = score, i
    top, bottom = " ".join(words[:cut]), " ".join(words[cut:])
    # A second line beginning with "[" would be swallowed as the optional vertical
    # space argument of \\, so it is shielded with an empty group.
    if bottom.startswith("["):
        bottom = "{}" + bottom
    return rf"\makecell[bc]{{{top}\\{bottom}}}"


def save_table(path: Path, columns, rows, alignment=None, caption=None, label=None):
    """Write one LaTeX table. Tables carry no notes; anything a reader needs in
    order to interpret a column is stated in the running text instead.

    The body is set at a fixed 8 pt rather than scaled to the text width. Scaling
    shrinks wide tables to whatever size makes them fit, which leaves the widest of
    them illegible at normal page size; a fixed size keeps every table readable and
    consistent with the others.
    """
    alignment = alignment or ("l" + "r" * (len(columns) - 1))
    lines = [r"\begin{table*}[!htbp]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines += [r"\fontsize{8}{9.6}\selectfont",
              r"\setlength{\tabcolsep}{2.5pt}",
              r"\setlength{\emergencystretch}{1.5em}",
              r"\sbox{\tabbox}{%",
              rf"\begin{{tabular}}{{{alignment}}}", r"\toprule",
              " & ".join(stacked_header(c) for c in columns) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              # Every table is set to exactly the text width. Narrow tables are
              # enlarged and wide ones reduced; the stacked headings and wrapped
              # text columns above keep the natural width close to the text block,
              # so the scaling stays mild in both directions.
              rf"\typeout{{TABLEWIDTH {label or path.stem}: \the\wd\tabbox\space of \the\textwidth}}",
              r"\resizebox{\textwidth}{!}{\usebox{\tabbox}}%"]
    lines.append(r"\end{table*}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifacts(results: dict, source: Path) -> None:
    """Write every machine-generated table and prose macro from the results record.

    The function depends only on the serialized results, so tables and macros can be
    regenerated from ``analysis_results.json`` without refitting any model.
    """
    all_results = results["models"]
    paired_differences = results["paired_differences"]
    cv_records = results["cv_selection"]
    data_audit = results["data_audit"]
    event_results = results["event_warning"]
    eligibility = results["event_eligibility"]
    locked_audit = results["locked_test_audit"]
    operating_conditions = results["operating_conditions"]
    inference_latency = results["inference_latency"]
    imbalance_study = results["imbalance_study"]
    boosting_study = results["boosting_implementations"]
    literature_study = results["literature_reimplementations"]
    recursive_study = results["recursive_multistep"]
    difficulty = results["configuration_difficulty"]
    importance_boosting = results["permutation_importance_boosting"]
    selected60 = all_results[60]["selected_learned_model"]
    shift_lookup = {x["feature"]: x["smd"] for x in results["distribution_shift"]}

    # Machine-generated LaTeX tables.
    data_rows = [
        ["Actual movements/runway use (TATIC TWR; restricted)", "2022-01-01--2024-06-30", "event", "after each completed 15-min bin", "no-operation state carried forward; mixed directions excluded", "state and labels"],
        ["Planned schedules (ANAC/SIROS; public)", "2022-01-01--2024-06-30", "event", "assumed available by issue time", "zero when no scheduled event; revision time unavailable", "target-bin planned demand"],
        ["METAR (REDEMET)", "2022-01-01--2024-06-30", "irregular", "latest report at or before issue time", "reports older than 120 min excluded", "weather"],
        ["Calendar", "full period", "15 min", "known", "complete by construction", "cyclic local-time terms"],
    ]
    save_table(source / "Table03_Data_Availability.tex",
               ["Source", "Coverage", "Native resolution", "Availability rule", "Completeness handling", "Use"],
               data_rows,
               alignment=(r">{\raggedright\arraybackslash}p{2.12cm}>{\raggedright\arraybackslash}p{1.85cm}>{\raggedright\arraybackslash}p{1.7cm}"
                          r">{\raggedright\arraybackslash}p{2.3cm}>{\raggedright\arraybackslash}p{2.2cm}>{\raggedright\arraybackslash}p{2.4cm}"),
               caption="Data sources and feature-availability rules.", label="tab:data")

    # Complete feature set for the three primary learned families.
    header = r"Group & Feature & Definition & Latest information used \\"
    columns = (r"{>{\raggedright\arraybackslash}p{2.05cm}"
               r">{\raggedright\arraybackslash}p{2.35cm}>{\raggedright\arraybackslash}X"
               r">{\raggedright\arraybackslash}p{2.45cm}}")
    feature_lines = [
        r"{\scriptsize",
        r"\begin{xltabular}{\textwidth}" + columns,
        rf"\caption{{Complete {len(FEATURE_DEFINITIONS)}-predictor feature set used by the three primary learned families.}}",
        r"\label{tab:features}\\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\multicolumn{4}{l}{\textit{Table \ref{tab:features} continued}}\\",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot",
    ]
    for group, members in FEATURE_GROUPS.items():
        for position, feature in enumerate(members):
            name_text, definition, availability = FEATURE_DEFINITIONS[feature]
            group_cell = GROUP_LABELS[group] if position == 0 else ""
            feature_lines.append(
                f"{group_cell} & {name_text} & {definition} & {availability} \\\\")
        feature_lines.append(r"\addlinespace")
    feature_lines += [r"\end{xltabular}", r"}"]
    (source / "Table04_Feature_Set.tex").write_text(
        "\n".join(feature_lines) + "\n", encoding="utf-8")

    main_rows = []
    for h in HORIZONS:
        for name in MODEL_LABELS:
            m = all_results[h][name]
            ci = m["ci"]["balanced_accuracy"]
            main_rows.append([str(h), MODEL_LABELS[name], f"{m['accuracy']:.3f}",
                              f"{m['balanced_accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
                              f"{m['macro_f1']:.3f}", f"{m['minority_f1']:.3f}",
                              f"{m['mcc']:.3f}", f"{m['brier']:.3f}"])
    save_table(source / "Table05_Main_Performance.tex",
               ["Horizon", "Method", "Accuracy", r"Balanced accuracy [95\% CI]",
                "Macro F1", "Minority F1", "MCC", "Brier"], main_rows,
               alignment="r l r r r r r r",
               caption="Locked-test performance by forecast horizon.", label="tab:main")

    # Transition-focused performance for every evaluated method at every horizon.
    transition_rows = []
    for h in HORIZONS:
        selected_name = all_results[h]["selected_learned_model"]
        for name in MODEL_LABELS:
            m = all_results[h][name]
            label = MODEL_LABELS[name]
            if name == selected_name:
                label += r"$^{\ast}$"
            transition_rows.append([
                str(h), label,
                f"{m['transition_precision']:.3f}", f"{m['transition_recall']:.3f}",
                f"{m['transition_f1']:.3f}", f"{m['missed_transition_intervals']:,}",
                f"{m['false_alerts_per_day']:.2f}",
                f"{m['warned_events']}", f"{m['event_recall']:.3f}"])
    save_table(source / "Table06_Transition_Performance.tex",
               ["Horizon", "Method", "Precision", "Recall", "F1", "Missed intervals",
                "False alerts/day", "Events warned", "Event recall"],
               transition_rows, alignment="r l r r r r r r r",
               caption=("Transition-focused performance of all seven primary methods on the locked "
                        "test period. An asterisk marks the model selected by rolling validation at "
                        f"that horizon. The two event columns are computed against the {eligibility['eligible_events']} "
                        "eligible consecutive-interval change events defined in "
                        "Section~\\ref{sec:transitions}, not "
                        "against every direction change in the test period."),
               label="tab:transition")

    condition_rows = []
    for h in HORIZONS:
        selected_name = all_results[h]["selected_learned_model"]
        for label, record in operating_conditions[h]["conditions"].items():
            selected_metrics = record["models"][selected_name]
            wind_metrics = record["models"]["wind_rule"]
            logistic_metrics = record["models"]["logistic"]
            persistence_metrics = record["models"]["persistence"]
            ci = selected_metrics["ci"]["balanced_accuracy"]
            condition_rows.append([
                str(h), label, f"{record['n']:,}", MODEL_LABELS[selected_name],
                f"{selected_metrics['balanced_accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
                f"{logistic_metrics['balanced_accuracy']:.3f}",
                f"{wind_metrics['balanced_accuracy']:.3f}",
                f"{persistence_metrics['balanced_accuracy']:.3f}",
                f"{selected_metrics['transition_recall']:.3f}",
                f"{selected_metrics['false_alerts_per_day']:.2f}",
            ])
    save_table(source / "Table10_Operating_Conditions.tex",
               ["Horizon", "Condition", "$n$", "Selected model",
                r"Balanced accuracy [95\% CI]", "Logistic", "Wind rule", "Persistence",
                "Transition recall", "False alerts/day"],
               condition_rows, alignment=r"r >{\raggedright\arraybackslash}p{2.2cm} r >{\raggedright\arraybackslash}p{1.45cm} r r r r r r",
               caption="Locked-test performance under operationally relevant conditions.",
               label="tab:conditions")

    # Boosting implementations under an identical selection and calibration protocol.
    boosting_rows = []
    for h in HORIZONS:
        for impl, point in boosting_study[h].items():
            ci = point["ci"]["balanced_accuracy"]
            boosting_rows.append([
                str(h), ALL_MODEL_LABELS[impl],
                f"{point['validation_balanced_accuracy']:.3f}",
                f"{point['balanced_accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
                f"{point['brier']:.3f}",
                f"{point['transition_recall']:.3f}", f"{point['transition_f1']:.3f}",
                f"{point['false_alerts_per_day']:.2f}",
                f"{point['fit_seconds']:.0f}"])
    save_table(source / "Table08_Boosting_Implementations.tex",
               ["Horizon", "Implementation", "Validation balanced accuracy",
                r"Locked-test balanced accuracy [95\% CI]", "Brier",
                "Transition recall", "Transition F1", "False alerts/day",
                "Fit time (s)"],
               boosting_rows, alignment=r"r >{\raggedright\arraybackslash}p{2.15cm} r r r r r r r",
               caption="Gradient-boosting implementations under an identical validation, calibration, and evaluation protocol.",
               label="tab:boosting")

    # Class-imbalance handling and alert-threshold criteria.
    imbalance_rows = []
    for h in HORIZONS:
        model_label = MODEL_LABELS[imbalance_study[h]["model"]]
        for label, point in imbalance_study[h]["strategies"].items():
            row = [
                str(h), model_label, label, f"{point['threshold']:.2f}",
                f"{point['balanced_accuracy']:.3f}",
                f"{point['transition_precision']:.3f}",
                f"{point['transition_recall']:.3f}", f"{point['transition_f1']:.3f}",
                f"{point['false_alerts_per_day']:.2f}",
                f"{point['event_recall']:.3f}"]
            previous = imbalance_rows[-1] if imbalance_rows else None
            # R2 presentation: when the alert-budget criterion selects the same threshold
            # as the transition-F1 criterion, the two identical rows are printed as one.
            if (previous is not None and previous[0] == row[0]
                    and previous[2].endswith("transition-F1 threshold")
                    and label.startswith("Class weighting, validation budget")
                    and previous[3:] == row[3:]):
                criterion = label.removeprefix("Class weighting, ")
                previous[2] += f"; the {criterion} selects the same threshold"
                continue
            imbalance_rows.append(row)
    save_table(source / "Table07_Imbalance_Handling.tex",
               ["Horizon", "Model", "Imbalance strategy and threshold criterion",
                "Threshold", "Balanced accuracy", "Transition precision",
                "Transition recall", "Transition F1", "False alerts/day",
                "Event recall"],
               imbalance_rows, alignment=r"r >{\raggedright\arraybackslash}p{2.2cm} >{\raggedright\arraybackslash}p{4.3cm} r r r r r r r",
               caption="Class-imbalance handling and alert-threshold criteria for the horizon-selected model.",
               label="tab:imbalance")

    # Comparators adapted from published designs.
    literature_rows = []
    for h in HORIZONS:
        selected_name = all_results[h]["selected_learned_model"]
        reference = all_results[h][selected_name]
        ci = reference["ci"]["balanced_accuracy"]
        literature_rows.append([
            str(h), "Direct multi-horizon classification (this study)",
            MODEL_LABELS[selected_name], f"{len(ALL_FEATURES)}",
            f"{reference['balanced_accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
            f"{reference['transition_recall']:.3f}", f"{reference['transition_f1']:.3f}",
            f"{reference['false_alerts_per_day']:.2f}"])
        dc = literature_study[h]["discrete_choice"]
        ci = dc["ci"]["balanced_accuracy"]
        literature_rows.append([
            str(h), r"Discrete-choice utility specification \citep{avery2015}",
            "Binary logit", f"{dc['n_features']}",
            f"{dc['balanced_accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
            f"{dc['transition_recall']:.3f}", f"{dc['transition_f1']:.3f}",
            f"{dc['false_alerts_per_day']:.2f}"])
        rec = recursive_study["horizons"][h]["recursive"]
        ci = rec["ci"]["balanced_accuracy"]
        literature_rows.append([
            str(h), r"Recursive multi-step prediction \citep{khater2021}",
            MODEL_LABELS[recursive_study["one_step_model"]] + ", one-step",
            f"{len(ALL_FEATURES)}",
            f"{rec['balanced_accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
            f"{rec['transition_recall']:.3f}",
            f"{rec['transition_f1']:.3f}",
            f"{rec['false_alerts_per_day']:.2f}"])
        mlp = literature_study[h]["mlp"]
        ci = mlp["ci"]["balanced_accuracy"]
        literature_rows.append([
            str(h), r"Comparative supervised benchmark \citep{raju2021,puranik2023}",
            "Neural network", f"{mlp['n_features']}",
            f"{mlp['balanced_accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
            f"{mlp['transition_recall']:.3f}", f"{mlp['transition_f1']:.3f}",
            f"{mlp['false_alerts_per_day']:.2f}"])
    save_table(source / "Table09_Literature_Reimplementation.tex",
               ["Horizon", "Published design", "SBGR implementation", "Predictors",
                r"Balanced accuracy [95\% CI]", "Transition recall", "Transition F1",
                "False alerts/day"],
               literature_rows, alignment=r"r >{\raggedright\arraybackslash}p{3.8cm} >{\raggedright\arraybackslash}p{3.4cm} r r r r r",
               caption="Comparators adapted from published designs, evaluated on the SBGR data under the study protocol. "
                       "Recursive rows use the common-step subset described in Section~\\ref{sec:comparison}; all other rows use the full locked test.",
               label="tab:literature")

    # Macros used to keep prose synchronized with the analysis.
    macros = [
        rf"\newcommand{{\DataStart}}{{{data_audit['start_label']}}}",
        rf"\newcommand{{\DataEnd}}{{{data_audit['end_label']}}}",
        rf"\newcommand{{\TotalIntervals}}{{{data_audit['intervals']:,}}}",
        rf"\newcommand{{\ObservedLabels}}{{{data_audit['observed_labels']:,}}}",
        rf"\newcommand{{\NoOpLabels}}{{{data_audit['no_operation_imputed']:,}}}",
        rf"\newcommand{{\MixedExcluded}}{{{data_audit['mixed_excluded']:,}}}",
        rf"\newcommand{{\TestRows}}{{{locked_audit['rows'][60]:,}}}",
        rf"\newcommand{{\TestObservedRows}}{{{locked_audit['observed_target_labels'][60]:,}}}",
        rf"\newcommand{{\TestLocalDates}}{{{locked_audit['local_operational_dates'][60]}}}",
        rf"\newcommand{{\TestEvents}}{{{event_results['events']}}}",
        rf"\newcommand{{\WarnedEvents}}{{{event_results['warned_within_tolerance']}}}",
        rf"\newcommand{{\MissedEvents}}{{{event_results['missed_events']}}}",
        rf"\newcommand{{\EventRecall}}{{{100 * event_results['event_recall']:.1f}\%}}",
        rf"\newcommand{{\MedianLead}}{{{event_results['median_warning_lead_minutes']:.0f}}}",
        rf"\newcommand{{\MedianTimingError}}{{{event_results['median_absolute_timing_error_minutes']:.0f}}}",
        rf"\newcommand{{\AllChanges}}{{{eligibility['all_direction_changes']}}}",
        rf"\newcommand{{\EligibleShare}}{{{100 * eligibility['eligible_share']:.1f}\%}}",
        rf"\newcommand{{\ExcludedChanges}}{{{eligibility['excluded_changes']}}}",
        rf"\newcommand{{\ExcludedOneMixed}}{{{eligibility['excluded_one_mixed_interval']}}}",
        rf"\newcommand{{\ExcludedTwoMixed}}{{{eligibility['excluded_two_mixed_intervals']}}}",
        rf"\newcommand{{\ChangesToTwentyEight}}{{{eligibility['all_changes_to_28']}}}",
        rf"\newcommand{{\ChangesToTen}}{{{eligibility['all_changes_to_10']}}}",
        rf"\newcommand{{\EligibleToTwentyEight}}{{{eligibility['eligible_to_28']}}}",
        rf"\newcommand{{\EligibleToTen}}{{{eligibility['eligible_to_10']}}}",
        rf"\newcommand{{\SelectedThresholdSixty}}{{{all_results[60][selected60]['selection_threshold']:.2f}}}",
        rf"\newcommand{{\InferenceLatencySixty}}{{{inference_latency[60]['median_milliseconds_per_1000_rows']:.0f}}}",
        rf"\newcommand{{\HighDemandBalSixty}}{{{operating_conditions[60]['conditions']['High planned demand']['models'][selected60]['balanced_accuracy']:.3f}}}",
        rf"\newcommand{{\WeatherBalSixty}}{{{operating_conditions[60]['conditions']['Weather-sensitive intervals']['models'][selected60]['balanced_accuracy']:.3f}}}",
        rf"\newcommand{{\HighDemandRecallSixty}}{{{operating_conditions[60]['conditions']['High planned demand']['models'][selected60]['transition_recall']:.3f}}}",
        rf"\newcommand{{\WeatherRecallSixty}}{{{operating_conditions[60]['conditions']['Weather-sensitive intervals']['models'][selected60]['transition_recall']:.3f}}}",
        rf"\newcommand{{\HighDemandAlertsSixty}}{{{operating_conditions[60]['conditions']['High planned demand']['models'][selected60]['false_alerts_per_day']:.2f}}}",
        rf"\newcommand{{\WeatherAlertsSixty}}{{{operating_conditions[60]['conditions']['Weather-sensitive intervals']['models'][selected60]['false_alerts_per_day']:.2f}}}",
        rf"\newcommand{{\HighDemandN}}{{{operating_conditions[60]['conditions']['High planned demand']['n']:,}}}",
        rf"\newcommand{{\WeatherN}}{{{operating_conditions[60]['conditions']['Weather-sensitive intervals']['n']:,}}}",
        rf"\newcommand{{\HighDemandLogisticSixty}}{{{operating_conditions[60]['conditions']['High planned demand']['models']['logistic']['balanced_accuracy']:.3f}}}",
        rf"\newcommand{{\HighDemandWindSixty}}{{{operating_conditions[60]['conditions']['High planned demand']['models']['wind_rule']['balanced_accuracy']:.3f}}}",
        rf"\newcommand{{\WeatherLogisticSixty}}{{{operating_conditions[60]['conditions']['Weather-sensitive intervals']['models']['logistic']['balanced_accuracy']:.3f}}}",
        rf"\newcommand{{\WeatherWindSixty}}{{{operating_conditions[60]['conditions']['Weather-sensitive intervals']['models']['wind_rule']['balanced_accuracy']:.3f}}}",
        rf"\newcommand{{\NumFeatures}}{{{len(ALL_FEATURES)}}}",
        rf"\newcommand{{\ChangesPerDay}}{{{difficulty['changes_per_local_day']:.2f}}}",
        rf"\newcommand{{\MedianRunHours}}{{{difficulty['median_run_hours']:.1f}}}",
        rf"\newcommand{{\MeanRunHours}}{{{difficulty['mean_run_hours']:.1f}}}",
        rf"\newcommand{{\DominantShare}}{{{100 * difficulty['dominant_configuration_share']:.1f}\%}}",
        rf"\newcommand{{\StateChangeShare}}{{{100 * difficulty['state_change_share_of_intervals']:.2f}\%}}",
        rf"\newcommand{{\NoChangeShare}}{{{100 * (1 - difficulty['state_change_share_of_intervals']):.2f}\%}}",
        rf"\newcommand{{\RecursiveCommonN}}{{{recursive_study['horizons'][60]['n']:,}}}",
        rf"\newcommand{{\HistBalSixty}}{{{boosting_study[60]['hist_gradient_boosting']['balanced_accuracy']:.3f}}}",
        rf"\newcommand{{\XGBoostTransitionFSixty}}{{{boosting_study[60]['xgboost']['transition_f1']:.3f}}}",
        rf"\newcommand{{\LightGBMTransitionFSixty}}{{{boosting_study[60]['lightgbm']['transition_f1']:.3f}}}",
    ]
    budget = imbalance_study[60]["strategies"]["Class weighting, validation budget of one alert per day"]
    transition_threshold = imbalance_study[60]["strategies"]["Class weighting, transition-F1 threshold"]
    undersample = imbalance_study[60]["strategies"]["Random undersampling"]
    oversample = imbalance_study[60]["strategies"]["Random oversampling"]
    smote = imbalance_study[60]["strategies"]["SMOTE"]
    unweighted = imbalance_study[60]["strategies"]["No reweighting or resampling"]
    macros += [
        rf"\newcommand{{\BudgetRecallSixty}}{{{budget['transition_recall']:.3f}}}",
        rf"\newcommand{{\BudgetAlertsSixty}}{{{budget['false_alerts_per_day']:.2f}}}",
        rf"\newcommand{{\BudgetThresholdSixty}}{{{budget['threshold']:.2f}}}",
        rf"\newcommand{{\TransitionThresholdRecallSixty}}{{{transition_threshold['transition_recall']:.3f}}}",
        rf"\newcommand{{\TransitionThresholdPrecisionSixty}}{{{transition_threshold['transition_precision']:.3f}}}",
        rf"\newcommand{{\TransitionThresholdAlertsSixty}}{{{transition_threshold['false_alerts_per_day']:.2f}}}",
        rf"\newcommand{{\TransitionThresholdFSixty}}{{{transition_threshold['transition_f1']:.3f}}}",
        rf"\newcommand{{\TransitionThresholdEventSixty}}{{{transition_threshold['event_recall']:.3f}}}",
        rf"\newcommand{{\TransitionThresholdValueSixty}}{{{transition_threshold['threshold']:.2f}}}",
        rf"\newcommand{{\UndersampleRecallSixty}}{{{undersample['transition_recall']:.3f}}}",
        rf"\newcommand{{\UndersampleAlertsSixty}}{{{undersample['false_alerts_per_day']:.2f}}}",
        rf"\newcommand{{\UndersampleFSixty}}{{{undersample['transition_f1']:.3f}}}",
        rf"\newcommand{{\UndersampleEventSixty}}{{{undersample['event_recall']:.3f}}}",
        rf"\newcommand{{\OversampleFSixty}}{{{oversample['transition_f1']:.3f}}}",
        rf"\newcommand{{\OversampleAlertsSixty}}{{{oversample['false_alerts_per_day']:.2f}}}",
        rf"\newcommand{{\SmoteAlertsSixty}}{{{smote['false_alerts_per_day']:.2f}}}",
        rf"\newcommand{{\OversampleEventSixty}}{{{oversample['event_recall']:.3f}}}",
        rf"\newcommand{{\SmoteFSixty}}{{{smote['transition_f1']:.3f}}}",
        rf"\newcommand{{\SmoteRecallSixty}}{{{smote['transition_recall']:.3f}}}",
        rf"\newcommand{{\SmoteEventSixty}}{{{smote['event_recall']:.3f}}}",
        rf"\newcommand{{\OversampleRecallSixty}}{{{oversample['transition_recall']:.3f}}}",
        rf"\newcommand{{\UnweightedEventSixty}}{{{unweighted['event_recall']:.3f}}}",
        rf"\newcommand{{\UnweightedAlertsSixty}}{{{unweighted['false_alerts_per_day']:.2f}}}",
        rf"\newcommand{{\UnweightedFSixty}}{{{unweighted['transition_f1']:.3f}}}",
        rf"\newcommand{{\RecursivePersistCommonSixty}}{{{recursive_study['horizons'][60]['persistence']['balanced_accuracy']:.3f}}}",
        rf"\newcommand{{\RecursiveLeadSixty}}{{{recursive_study['horizons'][60]['recursive']['median_warning_lead_minutes']:.1f}}}",
        rf"\newcommand{{\ReportedLeadSixty}}{{{imbalance_study[60]['strategies']['Class weighting (reported)']['median_warning_lead_minutes']:.0f}}}",
        rf"\newcommand{{\TransitionThresholdLeadSixty}}{{{transition_threshold['median_warning_lead_minutes']:.1f}}}",
        rf"\newcommand{{\TransitionThresholdBalSixty}}{{{transition_threshold['balanced_accuracy']:.3f}}}",
        rf"\newcommand{{\MajorityAcc}}{{{all_results[60]['majority']['accuracy']:.3f}}}",
        rf"\newcommand{{\MajorityBal}}{{{all_results[60]['majority']['balanced_accuracy']:.3f}}}",
        rf"\newcommand{{\EligibleForecasts}}{{{event_results['eligible_forecasts_assigned']}}}",
        rf"\newcommand{{\DuplicateEvents}}{{{event_results['events_with_duplicate_forecasts']}}}",
        rf"\newcommand{{\HighDemandShare}}{{{100 * operating_conditions[60]['conditions']['High planned demand']['share_of_test']:.1f}\%}}",
        rf"\newcommand{{\WeatherShare}}{{{100 * operating_conditions[60]['conditions']['Weather-sensitive intervals']['share_of_test']:.1f}\%}}",
        rf"\newcommand{{\HighDemandAlertShare}}{{{100 * operating_conditions[60]['conditions']['High planned demand']['models'][selected60]['false_alerts_per_day'] / all_results[60][selected60]['false_alerts_per_day']:.1f}\%}}",
        rf"\newcommand{{\WeatherAlertShare}}{{{100 * operating_conditions[60]['conditions']['Weather-sensitive intervals']['models'][selected60]['false_alerts_per_day'] / all_results[60][selected60]['false_alerts_per_day']:.1f}\%}}",
        rf"\newcommand{{\ShiftDoy}}{{{shift_lookup['target_doy_sin']:.2f}}}",
        rf"\newcommand{{\ShiftDewPoint}}{{{shift_lookup['dew_point']:.2f}}}",
        rf"\newcommand{{\ShiftTemperature}}{{{shift_lookup['temperature']:.2f}}}",
        rf"\newcommand{{\ShiftSmallCount}}{{{sum(1 for x in results['distribution_shift'] if abs(x['smd']) < 0.15)}}}",
    ]
    ablation_macro_names = {
        "Current state only": "AblationState",
        "Wind only": "AblationWind",
        "Wind + weather": "AblationWeather",
        "+ calendar": "AblationCalendar",
        "+ planned demand": "AblationDemand",
        "+ recent operations": "AblationRecentOps",
        "Complete (+ current state)": "AblationComplete",
    }
    for label, macro in ablation_macro_names.items():
        macros.append(
            rf"\newcommand{{\{macro}}}{{{results['ablation_60min'][label]['balanced_accuracy']:.3f}}}")
    for point in results["threshold_tradeoff_60min"]:
        name = {0.10: "Low", 0.50: "Mid"}.get(round(point["threshold"], 2))
        if name:
            macros += [
                rf"\newcommand{{\Threshold{name}Event}}{{{point['event_recall']:.3f}}}",
                rf"\newcommand{{\Threshold{name}Alerts}}{{{point['false_alerts_per_day']:.2f}}}",
            ]
    macros += [
        rf"\newcommand{{\WindEventRecallSixty}}{{{all_results[60]['wind_rule']['event_recall']:.3f}}}",
    ]
    for h in HORIZONS:
        selected_name = all_results[h]["selected_learned_model"]
        s = all_results[h][selected_name]
        p = all_results[h]["persistence"]
        prefix = {15: "Fifteen", 30: "Thirty", 60: "Sixty"}[h]
        macros += [
            rf"\newcommand{{\BestModel{prefix}}}{{{latex_escape(MODEL_LABELS[selected_name])}}}",
            rf"\newcommand{{\BestBal{prefix}}}{{{s['balanced_accuracy']:.3f}}}",
            rf"\newcommand{{\PersistBal{prefix}}}{{{p['balanced_accuracy']:.3f}}}",
            rf"\newcommand{{\BestTransitionF{prefix}}}{{{s['transition_f1']:.3f}}}",
        ]
        for comparator, comparison_prefix in [("persistence", "Persistence"), ("wind_rule", "Wind")]:
            comparison = paired_differences[h][selected_name][comparator]
            macros += [
                rf"\newcommand{{\Gain{comparison_prefix}{prefix}}}{{{comparison['difference']:.3f}}}",
                rf"\newcommand{{\Gain{comparison_prefix}{prefix}Low}}{{{comparison['ci'][0]:.3f}}}",
                rf"\newcommand{{\Gain{comparison_prefix}{prefix}High}}{{{comparison['ci'][1]:.3f}}}",
            ]
        macros += [
            rf"\newcommand{{\TransitionRecall{prefix}}}{{{s['transition_recall']:.3f}}}",
            rf"\newcommand{{\TransitionPrecision{prefix}}}{{{s['transition_precision']:.3f}}}",
            rf"\newcommand{{\FalseAlerts{prefix}}}{{{s['false_alerts_per_day']:.2f}}}",
            rf"\newcommand{{\MissedIntervals{prefix}}}{{{s['missed_transition_intervals']:,}}}",
            rf"\newcommand{{\WindTransitionF{prefix}}}{{{all_results[h]['wind_rule']['transition_f1']:.3f}}}",
            rf"\newcommand{{\WindTransitionRecall{prefix}}}{{{all_results[h]['wind_rule']['transition_recall']:.3f}}}",
            rf"\newcommand{{\WindTransitionPrecision{prefix}}}{{{all_results[h]['wind_rule']['transition_precision']:.3f}}}",
            rf"\newcommand{{\WindFalseAlerts{prefix}}}{{{all_results[h]['wind_rule']['false_alerts_per_day']:.2f}}}",
            rf"\newcommand{{\EventRecallHorizon{prefix}}}{{{s['event_recall']:.3f}}}",
        ]
        family_macro = {"logistic": "Logistic", "random_forest": "Forest",
                        "hist_gradient_boosting": "Boost"}
        for family, tag in family_macro.items():
            macros += [
                rf"\newcommand{{\{tag}Bal{prefix}}}{{{all_results[h][family]['balanced_accuracy']:.3f}}}",
                rf"\newcommand{{\Validation{tag}{prefix}}}{{{cv_records[h][family]['mean_balanced_accuracy']:.3f}}}",
                rf"\newcommand{{\{tag}TransitionF{prefix}}}{{{all_results[h][family]['transition_f1']:.3f}}}",
                rf"\newcommand{{\{tag}EventRecall{prefix}}}{{{all_results[h][family]['event_recall']:.3f}}}",
            ]
        macros += [
            rf"\newcommand{{\ObservedOnlyBal{prefix}}}{{{results['sensitivity'][h]['balanced_accuracy']:.3f}}}",
            rf"\newcommand{{\Brier{prefix}}}{{{s['brier']:.3f}}}",
            rf"\newcommand{{\ECE{prefix}}}{{{s['ece']:.3f}}}",
            rf"\newcommand{{\BestBalLow{prefix}}}{{{s['ci']['balanced_accuracy'][0]:.3f}}}",
            rf"\newcommand{{\BestBalHigh{prefix}}}{{{s['ci']['balanced_accuracy'][1]:.3f}}}",
            rf"\newcommand{{\BestAcc{prefix}}}{{{s['accuracy']:.3f}}}",
            rf"\newcommand{{\PersistAcc{prefix}}}{{{p['accuracy']:.3f}}}",
            rf"\newcommand{{\TransitionIntervals{prefix}}}{{{s['transition_intervals']:,}}}",
            rf"\newcommand{{\DiscreteChoiceBal{prefix}}}{{{literature_study[h]['discrete_choice']['balanced_accuracy']:.3f}}}",
            rf"\newcommand{{\DiscreteChoiceTransitionF{prefix}}}{{{literature_study[h]['discrete_choice']['transition_f1']:.3f}}}",
            rf"\newcommand{{\NeuralBal{prefix}}}{{{literature_study[h]['mlp']['balanced_accuracy']:.3f}}}",
            rf"\newcommand{{\XGBoostBal{prefix}}}{{{boosting_study[h]['xgboost']['balanced_accuracy']:.3f}}}",
            rf"\newcommand{{\LightGBMBal{prefix}}}{{{boosting_study[h]['lightgbm']['balanced_accuracy']:.3f}}}",
            rf"\newcommand{{\XGBoostTransitionRecall{prefix}}}{{{boosting_study[h]['xgboost']['transition_recall']:.3f}}}",
            rf"\newcommand{{\LightGBMTransitionRecall{prefix}}}{{{boosting_study[h]['lightgbm']['transition_recall']:.3f}}}",
            rf"\newcommand{{\RecursiveBal{prefix}}}{{{recursive_study['horizons'][h]['recursive']['balanced_accuracy']:.3f}}}",
            rf"\newcommand{{\DirectCommonBal{prefix}}}{{{recursive_study['horizons'][h]['direct']['balanced_accuracy']:.3f}}}",
            rf"\newcommand{{\RecursiveTransitionF{prefix}}}{{{recursive_study['horizons'][h]['recursive']['transition_f1']:.3f}}}",
            rf"\newcommand{{\DirectCommonTransitionF{prefix}}}{{{recursive_study['horizons'][h]['direct']['transition_f1']:.3f}}}",
            rf"\newcommand{{\MajorityTransitionF{prefix}}}{{{all_results[h]['majority']['transition_f1']:.3f}}}",
            rf"\newcommand{{\ClimatologyTransitionF{prefix}}}{{{all_results[h]['climatology']['transition_f1']:.3f}}}",
            rf"\newcommand{{\ClimatologyEventRecall{prefix}}}{{{all_results[h]['climatology']['event_recall']:.3f}}}",
            rf"\newcommand{{\ClimatologyAlerts{prefix}}}{{{all_results[h]['climatology']['false_alerts_per_day']:.2f}}}",
            rf"\newcommand{{\ValidationBal{prefix}}}{{{cv_records[h][selected_name]['mean_balanced_accuracy']:.3f}}}",
        ]
        ordinal = ("A", "B", "C")
        for rank, entry in enumerate(importance_boosting[h][:3]):
            label = FEATURE_DEFINITIONS[entry["feature"]][0]
            macros += [
                rf"\newcommand{{\TopFeature{ordinal[rank]}{prefix}}}{{{label.lower()}}}",
                rf"\newcommand{{\TopImportance{ordinal[rank]}{prefix}}}{{{entry['mean']:.3f}}}",
            ]
    (source / "generated_results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")



def analyze(data_dir: Path, out: Path):
    source = out
    analysis_archive = out
    figures = out
    figures.mkdir(parents=True, exist_ok=True)
    base = add_issue_features(build_interval_data(data_dir))

    data_audit = {
        "start": str(base["time"].min()),
        "end": str(base["time"].max()),
        "start_label": f"{base['time'].min().day} {base['time'].min():%B %Y}",
        "end_label": f"{base['time'].max().day} {base['time'].max():%B %Y}",
        "intervals": int(len(base)),
        "observed_labels": int((base["primary_label_source"] == "observed").sum()),
        "no_operation_imputed": int((base["primary_label_source"] == "carried_forward_no_operations").sum()),
        "mixed_excluded": int((base["primary_label_source"] == "mixed_excluded").sum()),
        "raw_metar_age_max_minutes": float(base["metar_age_minutes"].max()),
        "accepted_metar_age_max_minutes": float(
            horizon_frame(base, HORIZONS[0])["metar_age_minutes"].max()
        ),
    }

    # Eligible consecutive-interval change events in the locked test period. Defined
    # before the horizon loop so that event-level warning coverage can be reported for
    # every method. An interval in which movements occurred in both directions has no
    # unambiguous configuration and is excluded from the label series, so a change
    # whose changeover interval is of that kind separates the two adjacent pure
    # intervals by more than one step and has no single observable change time. Such
    # changes are outside the event-level estimand, although they remain inside the
    # interval-level transition universe, which uses the carried-forward state.
    state = base[["time", "primary_configuration"]].dropna().copy()
    state["prev_time"] = state["time"].shift()
    state["prev_cfg"] = state["primary_configuration"].shift()
    in_window = state["time"].between(TEST_START, TEST_END)
    changed = state["primary_configuration"].ne(state["prev_cfg"]) & state["prev_cfg"].notna()
    adjacent = state["time"] - state["prev_time"] == pd.Timedelta(minutes=15)
    events = state[in_window & changed & adjacent].copy()
    excluded_changes = state[in_window & changed & ~adjacent].copy()

    # Ledger reconciling the event denominator with every direction change in the
    # carried-forward state sequence over the same period.
    carried = base[["time", "state_configuration"]].copy()
    carried["prev_state"] = carried["state_configuration"].shift()
    all_changes = carried[carried["time"].between(TEST_START, TEST_END)
                          & carried["state_configuration"].notna()
                          & carried["prev_state"].notna()
                          & carried["state_configuration"].ne(carried["prev_state"])]
    gaps = ((excluded_changes["time"] - excluded_changes["prev_time"])
            .dt.total_seconds().div(60).astype(int))
    event_eligibility = {
        "all_direction_changes": int(len(all_changes)),
        "eligible_events": int(len(events)),
        "excluded_changes": int(len(excluded_changes)),
        "excluded_one_mixed_interval": int((gaps == 30).sum()),
        "excluded_two_mixed_intervals": int((gaps == 45).sum()),
        "eligible_share": float(len(events) / len(all_changes)) if len(all_changes) else 0.0,
        "all_changes_to_28": int((all_changes["state_configuration"] == "28L|28R").sum()),
        "all_changes_to_10": int((all_changes["state_configuration"] == "10R|10L").sum()),
        "eligible_to_28": int((events["primary_configuration"] == "28L|28R").sum()),
        "eligible_to_10": int((events["primary_configuration"] == "10R|10L").sum()),
    }

    all_results = {}
    fitted = {}
    calibrators = {}
    test_frames = {}
    cv_records = {}
    predictions = {}
    paired_differences = {}

    for horizon in HORIZONS:
        frame = horizon_frame(base, horizon)
        dev, test = dev_test_split(frame)
        test_frames[horizon] = test
        result_h = {}
        pred_h = {}
        model_selection = {}

        for name in MODEL_GRIDS:
            best, records = select_hyperparameters(dev, ALL_FEATURES, name)
            model_selection[name] = best
            final, cal, threshold, prob, oof = fitted_model_with_calibration(
                dev, test, ALL_FEATURES, name, best["params"])
            pred = (prob >= threshold).astype(int)
            pred_h[name] = {"pred": pred, "prob": prob, "threshold": threshold}
            fitted[(horizon, name)] = final
            calibrators[(horizon, name)] = cal
        cv_records[horizon] = model_selection

        y_test = test["y"].to_numpy()
        prevalence = dev["y"].mean()
        majority_pred = np.full(len(test), int(prevalence >= .5), dtype=int)
        pred_h["majority"] = {"pred": majority_pred,
                              "prob": np.full(len(test), prevalence), "threshold": .5}
        current = test["current_is_28"].astype(int).to_numpy()
        pred_h["persistence"] = {"pred": current, "prob": current.astype(float), "threshold": .5}
        clim_prob = fit_climatology(dev, test)
        pred_h["climatology"] = {"pred": (clim_prob >= .5).astype(int),
                                  "prob": clim_prob, "threshold": .5}
        validation_2023 = dev[dev["target_time"].dt.year == 2023]
        margins = [0.0, 2.0, 4.0, 6.0, 8.0]
        margin = max(margins, key=lambda m: balanced_accuracy_score(
            validation_2023["y"], wind_prediction(validation_2023, m)))
        wind_pred = wind_prediction(test, margin)
        pred_h["wind_rule"] = {"pred": wind_pred, "prob": wind_pred.astype(float),
                               "threshold": margin}

        for name, obj in pred_h.items():
            point = metrics(y_test, obj["pred"], obj["prob"])
            point["selection_threshold"] = float(obj["threshold"])
            ci, samples = block_bootstrap(test, obj["pred"], obj["prob"])
            point["ci"] = ci
            point.update(transition_metrics(test, obj["pred"]))
            point["confusion_matrix"] = confusion_matrix(y_test, obj["pred"], labels=[0, 1]).tolist()
            # Horizon-specific event coverage, so that warning value can be compared
            # across every evaluated method and not only for the selected model.
            single = event_warning_metrics(events, [(horizon, test, obj["pred"])],
                                           tolerance_minutes=15)
            point["event_recall"] = single["event_recall"]
            point["warned_events"] = single["warned_within_tolerance"]
            point["median_warning_lead_minutes"] = single["median_warning_lead_minutes"]
            result_h[name] = point

        # Persistence and the wind rule are pre-specified operational comparators.
        # Both are reported so the locked test is never used to choose a comparator.
        comparators = ["persistence", "wind_rule"]
        _, samples = block_bootstrap(
            test, pred_h["persistence"]["pred"], pred_h["persistence"]["prob"]
        )
        paired_h = {}
        for name in MODEL_GRIDS:
            paired_h[name] = {}
            for comparator in comparators:
                diffs = []
                y = y_test
                for idx in samples:
                    a = balanced_accuracy_score(y[idx], pred_h[name]["pred"][idx])
                    b = balanced_accuracy_score(y[idx], pred_h[comparator]["pred"][idx])
                    diffs.append(a - b)
                paired_h[name][comparator] = {
                    "difference": (
                        result_h[name]["balanced_accuracy"]
                        - result_h[comparator]["balanced_accuracy"]
                    ),
                    "ci": [float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))],
                }
        paired_differences[horizon] = paired_h
        learned = max(MODEL_GRIDS, key=lambda n: model_selection[n]["mean_balanced_accuracy"])
        result_h["selected_learned_model"] = learned
        all_results[horizon] = result_h
        predictions[horizon] = pred_h

    # Label sensitivity: evaluate the selected learned model on observed targets only.
    sensitivity = {}
    for horizon in HORIZONS:
        test = test_frames[horizon]
        mask = test["target_label_source"].eq("observed").to_numpy()
        selected = all_results[horizon]["selected_learned_model"]
        obj = predictions[horizon][selected]
        sensitivity[horizon] = {
            "observed_only_n": int(mask.sum()),
            "balanced_accuracy": balanced_accuracy_score(test.loc[mask, "y"], obj["pred"][mask]),
            "persistence_balanced_accuracy": balanced_accuracy_score(
                test.loc[mask, "y"], predictions[horizon]["persistence"]["pred"][mask]),
        }

    # Operational-condition evaluation. All subgroup cutoffs are fixed from development data.
    operating_conditions = {}
    for horizon in HORIZONS:
        frame = horizon_frame(base, horizon)
        dev = development_split(frame)
        test = test_frames[horizon]
        dev_demand = dev["target_scheduled_arrivals"] + dev["target_scheduled_departures"]
        test_demand = test["target_scheduled_arrivals"] + test["target_scheduled_departures"]
        cutoffs = {
            "planned_demand_q75": float(dev_demand.quantile(.75)),
            "wind_speed_q75": float(dev["wind_speed"].quantile(.75)),
            "visibility_q25": float(dev["visibility"].quantile(.25)),
        }
        challenging_weather = (
            test["wind_speed"].ge(cutoffs["wind_speed_q75"])
            | test["visibility"].lt(cutoffs["visibility_q25"])
            | test[["ifr", "lifr", "ts"]].fillna(0).gt(0).any(axis=1)
        )
        masks = OrderedDict([
            ("All intervals", np.ones(len(test), dtype=bool)),
            ("High planned demand", test_demand.ge(cutoffs["planned_demand_q75"]).to_numpy()),
            ("Weather-sensitive intervals", challenging_weather.to_numpy()),
        ])
        selected_name = all_results[horizon]["selected_learned_model"]
        condition_results = {}
        for label, mask in masks.items():
            subset = test.loc[mask].reset_index(drop=True)
            models = {}
            for name in ["persistence", "wind_rule", "logistic", selected_name]:
                obj = predictions[horizon][name]
                point = metrics(subset["y"], obj["pred"][mask], obj["prob"][mask])
                point.update(transition_metrics(subset, obj["pred"][mask]))
                if name == selected_name:
                    point["ci"], _ = block_bootstrap(
                        subset, obj["pred"][mask], obj["prob"][mask], n_boot=500
                    )
                models[name] = point
            condition_results[label] = {
                "n": int(mask.sum()),
                "share_of_test": float(mask.mean()),
                "models": models,
            }
        operating_conditions[horizon] = {
            "development_cutoffs": cutoffs,
            "conditions": condition_results,
        }

    # Model-only prediction latency, excluding upstream data-feed and feature-engineering delays.
    inference_latency = {}
    for horizon in HORIZONS:
        test = test_frames[horizon]
        selected_name = all_results[horizon]["selected_learned_model"]
        model = fitted[(horizon, selected_name)]
        cal = calibrators[(horizon, selected_name)]
        elapsed = []
        for _ in range(7):
            start = time.perf_counter()
            raw = model.predict_proba(test[ALL_FEATURES])[:, 1]
            apply_calibrator(cal, raw)
            elapsed.append(time.perf_counter() - start)
        inference_latency[horizon] = {
            "model": selected_name,
            "test_rows": int(len(test)),
            "median_total_seconds": float(np.median(elapsed)),
            "median_milliseconds_per_1000_rows": float(
                1000 * np.median(elapsed) * 1000 / len(test)
            ),
        }

    # Feature-group ablation for the 60-minute selected model.
    horizon = 60
    frame = horizon_frame(base, horizon)
    dev = development_split(frame)
    test = test_frames[horizon]
    selected = all_results[horizon]["selected_learned_model"]
    params = cv_records[horizon][selected]["params"]
    ablation_sets = OrderedDict([
        ("Current state only", FEATURE_GROUPS["current"]),
        ("Wind only", FEATURE_GROUPS["wind"]),
        ("Wind + weather", FEATURE_GROUPS["wind"] + FEATURE_GROUPS["weather"]),
        ("+ calendar", FEATURE_GROUPS["wind"] + FEATURE_GROUPS["weather"] + FEATURE_GROUPS["calendar"]),
        ("+ planned demand", FEATURE_GROUPS["wind"] + FEATURE_GROUPS["weather"] + FEATURE_GROUPS["calendar"] + FEATURE_GROUPS["planned"]),
        ("+ recent operations", FEATURE_GROUPS["wind"] + FEATURE_GROUPS["weather"] + FEATURE_GROUPS["calendar"] + FEATURE_GROUPS["planned"] + FEATURE_GROUPS["recent_ops"]),
        ("Complete (+ current state)", ALL_FEATURES),
    ])
    ablation = {}
    for label, features in ablation_sets.items():
        model, cal, threshold, prob, _ = fitted_model_with_calibration(
            dev, test, features, selected, params
        )
        pred = (prob >= threshold).astype(int)
        ci, _ = block_bootstrap(test, pred, prob, n_boot=500)
        ablation[label] = metrics(test["y"], pred, prob)
        ablation[label]["ci"] = ci
        ablation[label]["selection_threshold"] = threshold

    # Permutation importance for the final calibrated decision rule at every horizon.
    # Importances are also computed for a single family across horizons, so that the
    # horizon comparison is not confounded by a change of model family.
    def permutation_profile(horizon: int, model_name: str):
        model = fitted[(horizon, model_name)]
        calibrator = calibrators[(horizon, model_name)]
        threshold = predictions[horizon][model_name]["threshold"]
        frame = test_frames[horizon]

        def calibrated_threshold_balanced_accuracy(fitted_model, x, y):
            raw = fitted_model.predict_proba(x)[:, 1]
            probability = apply_calibrator(calibrator, raw)
            return balanced_accuracy_score(y, (probability >= threshold).astype(int))

        perm = permutation_importance(model, frame[ALL_FEATURES], frame["y"],
                                      scoring=calibrated_threshold_balanced_accuracy,
                                      n_repeats=5, random_state=SEED, n_jobs=1)
        return sorted([
            {"feature": f, "mean": float(m), "std": float(s)}
            for f, m, s in zip(ALL_FEATURES, perm.importances_mean, perm.importances_std)
        ], key=lambda x: x["mean"], reverse=True)

    importance_selected = {h: permutation_profile(h, all_results[h]["selected_learned_model"])
                           for h in HORIZONS}
    importance_boosting = {h: permutation_profile(h, "hist_gradient_boosting")
                           for h in HORIZONS}
    importance = importance_selected[60]

    # Temporal shift evidence via standardized mean differences.
    shift = []
    for feature in ALL_FEATURES:
        a, b = dev[feature].astype(float), test[feature].astype(float)
        pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0
        shift.append({"feature": feature, "smd": float(smd)})
    shift.sort(key=lambda x: abs(x["smd"]), reverse=True)

    # Event-level warning analysis using all selected horizon models.
    candidate_sets = []
    for h in HORIZONS:
        tf = test_frames[h]
        name = all_results[h]["selected_learned_model"]
        candidate_sets.append((h, tf, predictions[h][name]["pred"]))
    event_results = event_warning_metrics(events, candidate_sets, tolerance_minutes=15)

    # Locked-test sensitivity curve: classification threshold versus warning burden.
    # The selected operating threshold remains the rolling-validation threshold.
    threshold_tradeoff_60min = []
    tf60 = test_frames[60]
    selected60 = all_results[60]["selected_learned_model"]
    prob60 = predictions[60][selected60]["prob"]
    for threshold in np.linspace(.10, .90, 17):
        pred = (prob60 >= threshold).astype(int)
        tm = transition_metrics(tf60, pred)
        em = event_warning_metrics(events, [(60, tf60, pred)], tolerance_minutes=15)
        threshold_tradeoff_60min.append({
            "threshold": float(threshold),
            "balanced_accuracy": float(balanced_accuracy_score(tf60["y"], pred)),
            "transition_precision": tm["transition_precision"],
            "transition_recall": tm["transition_recall"],
            "false_alerts_per_day": tm["false_alerts_per_day"],
            "event_recall": em["event_recall"],
            "missed_events": em["missed_events"],
            "median_warning_lead_minutes": em["median_warning_lead_minutes"],
        })

    def development_frame(horizon: int):
        return development_split(horizon_frame(base, horizon))

    # Class-imbalance handling: reweighting, resampling, and alternative thresholds.
    # Resampling is confined to training folds and all thresholds are selected from
    # rolling out-of-fold predictions, so the locked test is never used for tuning.
    imbalance_grid = np.linspace(0.05, 0.95, 181)
    imbalance_study = {}
    for h in HORIZONS:
        dev_h = development_frame(h)
        test_h = test_frames[h]
        name_h = all_results[h]["selected_learned_model"]
        params_h = cv_records[h][name_h]["params"]
        strategies = OrderedDict()
        obj = predictions[h][name_h]
        reported = summarize_predictions(test_h, obj["pred"], obj["prob"],
                                         events=events, horizon=h)
        reported["threshold"] = float(obj["threshold"])
        strategies["Class weighting (reported)"] = reported
        for label, balanced, resampling in [
            ("No reweighting or resampling", False, "none"),
            ("Random oversampling", False, "oversample"),
            ("Random undersampling", False, "undersample"),
            ("SMOTE", False, "smote"),
        ]:
            _, _, thr, prob, _ = fitted_model_with_calibration(
                dev_h, test_h, ALL_FEATURES, name_h, params_h,
                balanced=balanced, resampling=resampling,
                threshold_criterion="balanced_accuracy", threshold_grid=imbalance_grid)
            pred = (prob >= thr).astype(int)
            entry = summarize_predictions(test_h, pred, prob, events=events, horizon=h)
            entry["threshold"] = thr
            strategies[label] = entry
        _, _, _, prob_w, oof_w = fitted_model_with_calibration(
            dev_h, test_h, ALL_FEATURES, name_h, params_h)
        for label, criterion in [
            ("Class weighting, transition-F1 threshold", "transition_f1"),
            ("Class weighting, validation budget of one alert per day", "alert_budget"),
        ]:
            thr = choose_threshold(oof_w, oof_w["prob"].to_numpy(), criterion,
                                   imbalance_grid, alert_budget=1.0)
            pred = (prob_w >= thr).astype(int)
            entry = summarize_predictions(test_h, pred, prob_w, events=events, horizon=h)
            entry["threshold"] = thr
            strategies[label] = entry
        imbalance_study[h] = {"model": name_h, "strategies": strategies}

    # Boosting implementations evaluated under one identical protocol.
    boosting_study = {}
    for h in HORIZONS:
        dev_h = development_frame(h)
        test_h = test_frames[h]
        entry = OrderedDict()
        for impl in ("hist_gradient_boosting", "xgboost", "lightgbm"):
            grid = MODEL_GRIDS.get(impl, EXTRA_GRIDS.get(impl))
            best, _ = select_hyperparameters(dev_h, ALL_FEATURES, impl, grid=grid)
            start = time.perf_counter()
            _, _, thr, prob, _ = fitted_model_with_calibration(
                dev_h, test_h, ALL_FEATURES, impl, best["params"])
            fit_seconds = time.perf_counter() - start
            pred = (prob >= thr).astype(int)
            point = summarize_predictions(test_h, pred, prob, events=events,
                                          horizon=h, n_boot=500)
            point["validation_balanced_accuracy"] = best["mean_balanced_accuracy"]
            point["fit_seconds"] = float(fit_seconds)
            point["threshold"] = thr
            entry[impl] = point
        boosting_study[h] = entry

    # Comparators adapted from published designs.
    literature_study = {}
    for h in HORIZONS:
        dev_h = development_frame(h)
        test_h = test_frames[h]
        entry = OrderedDict()
        for impl, feats in [("discrete_choice", DISCRETE_CHOICE_FEATURES),
                            ("mlp", ALL_FEATURES)]:
            best, _ = select_hyperparameters(dev_h, feats, impl, grid=EXTRA_GRIDS[impl])
            _, _, thr, prob, _ = fitted_model_with_calibration(
                dev_h, test_h, feats, impl, best["params"])
            pred = (prob >= thr).astype(int)
            point = summarize_predictions(test_h, pred, prob, events=events,
                                          horizon=h, n_boot=500)
            point["validation_balanced_accuracy"] = best["mean_balanced_accuracy"]
            point["threshold"] = thr
            point["n_features"] = len(feats)
            entry[impl] = point
        literature_study[h] = entry

    # Recursive multi-step forecasting from a single one-step classifier.
    rec_name = all_results[15]["selected_learned_model"]
    rec_params = cv_records[15][rec_name]["params"]
    rec_frames, rec_preds, rec_threshold = recursive_multistep(
        base, ALL_FEATURES, rec_name, rec_params)
    recursive_study = {"one_step_model": rec_name, "threshold": rec_threshold,
                       "horizons": {}}
    for k in HORIZONS:
        fr = rec_frames[k]
        rec_point = summarize_predictions(fr, rec_preds[k]["pred"], rec_preds[k]["prob"],
                                          events=events, horizon=k, n_boot=500)
        direct_name = all_results[k]["selected_learned_model"]
        tf = test_frames[k]
        mask = tf["issue_time"].isin(set(fr["issue_time"])).to_numpy()
        direct_frame = tf.loc[mask].reset_index(drop=True)
        direct_point = summarize_predictions(
            direct_frame, predictions[k][direct_name]["pred"][mask],
            predictions[k][direct_name]["prob"][mask], events=events, horizon=k,
            n_boot=500)
        persistence_point = summarize_predictions(
            direct_frame, predictions[k]["persistence"]["pred"][mask],
            predictions[k]["persistence"]["prob"][mask], events=events, horizon=k,
            n_boot=500)
        recursive_study["horizons"][k] = {
            "n": int(len(fr)), "recursive": rec_point, "direct": direct_point,
            "persistence": persistence_point, "direct_model": direct_name,
        }

    # Structural difficulty of the SBGR configuration process.
    states = base["state_configuration"]
    change_flags = states.ne(states.shift()) & states.notna() & states.shift().notna()
    run_id = change_flags.cumsum()
    run_lengths = states.groupby(run_id).size() * 0.25
    local_dates = base["time"].dt.tz_convert("America/Sao_Paulo").dt.date
    difficulty = {
        "distinct_configurations": int(states.dropna().nunique()),
        "dominant_configuration_share": float((states == "10R|10L").mean()),
        "state_change_share_of_intervals": float(change_flags.mean()),
        "changes_per_local_day": float(change_flags.sum() / max(1, pd.Series(local_dates).nunique())),
        "median_run_hours": float(run_lengths.median()),
        "mean_run_hours": float(run_lengths.mean()),
        "p90_run_hours": float(run_lengths.quantile(0.90)),
        "persistence_balanced_accuracy": {
            h: all_results[h]["persistence"]["balanced_accuracy"] for h in HORIZONS},
        "majority_accuracy": float(all_results[60]["majority"]["accuracy"]),
    }

    # Monthly dynamics and monthly performance.
    dynamics = base.assign(month=base["time"].dt.tz_localize(None).dt.to_period("M")).groupby("month").agg(
        intervals=("time", "size"),
        p28=("state_configuration", lambda s: (s == "28L|28R").mean()),
        mixed=("mixed_directions", "sum"),
    ).reset_index()
    changes = base["state_configuration"].ne(base["state_configuration"].shift()).astype(int)
    dynamics["transition_rate"] = (base.assign(month=base["time"].dt.tz_localize(None).dt.to_period("M"), change=changes)
                                     .groupby("month")["change"].mean().to_numpy())
    monthly_perf = []
    for h in HORIZONS:
        tf = test_frames[h].copy()
        tf["month"] = tf["target_time"].dt.tz_localize(None).dt.to_period("M")
        name = all_results[h]["selected_learned_model"]
        for month, idx in tf.groupby("month").groups.items():
            idx = np.asarray(list(idx))
            if tf.iloc[idx]["y"].nunique() < 2:
                continue
            monthly_perf.append({
                "horizon": h, "month": str(month),
                "selected": balanced_accuracy_score(tf.iloc[idx]["y"], predictions[h][name]["pred"][idx]),
                "persistence": balanced_accuracy_score(tf.iloc[idx]["y"], predictions[h]["persistence"]["pred"][idx]),
            })

    # Locked-test composition. The three horizon-specific test sets happen to contain
    # the same number of rows, but they are built from different issue times, so they
    # are not a common set of cases and are not described as one.
    locked_test_audit = {
        "rows": {h: int(len(test_frames[h])) for h in HORIZONS},
        "observed_target_labels": {
            h: int((test_frames[h]["target_label_source"] == "observed").sum()) for h in HORIZONS},
        "shared_issue_times": int(len(set.intersection(
            *[set(test_frames[h]["issue_time"]) for h in HORIZONS]))),
        "shared_target_times": int(len(set.intersection(
            *[set(test_frames[h]["target_time"]) for h in HORIZONS]))),
        "local_operational_dates": {h: int(test_frames[h]["target_date"].nunique()) for h in HORIZONS},
    }

    results = {
        "data_audit": data_audit,
        "locked_test_audit": locked_test_audit,
        "models": all_results,
        "paired_differences": paired_differences,
        "cv_selection": cv_records,
        "sensitivity": sensitivity,
        "ablation_60min": ablation,
        "permutation_importance_60min": importance,
        "permutation_importance_selected": importance_selected,
        "permutation_importance_boosting": importance_boosting,
        "imbalance_study": imbalance_study,
        "boosting_implementations": boosting_study,
        "literature_reimplementations": literature_study,
        "recursive_multistep": recursive_study,
        "configuration_difficulty": difficulty,
        "distribution_shift": shift,
        "event_warning": event_results,
        "event_eligibility": event_eligibility,
        "threshold_tradeoff_60min": threshold_tradeoff_60min,
        "operating_conditions": operating_conditions,
        "inference_latency": inference_latency,
        "software": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "pandas": pd.__version__, "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "matplotlib": mpl.__version__,
            "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
            "random_seed": SEED,
        },
    }
    analysis_archive.mkdir(parents=True, exist_ok=True)
    (analysis_archive / "analysis_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    write_artifacts(results, source)

    # Plotted series that the results record does not hold. They are written next to
    # it so that make_figures.py can redraw every figure without refitting.
    wind_advantage = base["headwind_28"] - base["headwind_10"]
    wind_bins = pd.cut(wind_advantage, [-np.inf, -10, -5, 0, 5, 10, np.inf])
    wind_profile = (base.assign(wind_bin=wind_bins)
                    .groupby("wind_bin", observed=True)["state_configuration"]
                    .apply(lambda s: (s == "28L|28R").mean()))
    tf = test_frames[60]; name = all_results[60]["selected_learned_model"]
    prob = predictions[60][name]["prob"]; y = tf["y"].to_numpy()
    bins = np.linspace(0, 1, 11); ids = np.clip(np.digitize(prob, bins, right=True)-1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        mask = ids == b
        if mask.any(): xs.append(prob[mask].mean()); ys.append(y[mask].mean()); ns.append(mask.sum())
    figure_data = {
        "configuration_dynamics": {
            "month": dynamics["month"].astype(str).tolist(),
            "p28": dynamics["p28"].astype(float).tolist(),
            "transition_rate": dynamics["transition_rate"].astype(float).tolist(),
        },
        "wind_profile": {
            "bins": ["<-10", "-10:-5", "-5:0", "0:5", "5:10", ">10"],
            "share_28": wind_profile.astype(float).tolist(),
        },
        "reliability_60min_selected": {
            "model": name,
            "mean_predicted": [float(v) for v in xs],
            "observed": [float(v) for v in ys],
            "n": [int(v) for v in ns],
        },
        "monthly_performance": [{k: (float(v) if k in ("selected", "persistence") else v)
                                 for k, v in m.items()} for m in monthly_perf],
    }
    (analysis_archive / "figure_data.json").write_text(
        json.dumps(figure_data, indent=2, default=str), encoding="utf-8")

    draw_all(json.loads(json.dumps(results, default=str)), figure_data, figures,
             feature_labels={k: v[0] for k, v in FEATURE_DEFINITIONS.items()},
             model_labels=dict(MODEL_LABELS))

    print(json.dumps({
        "data_audit": data_audit,
        "selected_models": {h: all_results[h]["selected_learned_model"] for h in HORIZONS},
        "event_warning": event_results,
        "balanced_accuracy": {h: {n: all_results[h][n]["balanced_accuracy"] for n in MODEL_LABELS}
                              for h in HORIZONS},
    }, indent=2))


def normalize_results(results: dict) -> dict:
    """Restore integer horizon keys lost to JSON serialization."""
    def to_int_keys(mapping):
        return {int(k): v for k, v in mapping.items()}

    for key in ("models", "paired_differences", "cv_selection", "sensitivity",
                "imbalance_study", "boosting_implementations",
                "literature_reimplementations", "operating_conditions",
                "inference_latency", "permutation_importance_selected",
                "permutation_importance_boosting"):
        if key in results:
            results[key] = to_int_keys(results[key])
    if "recursive_multistep" in results:
        results["recursive_multistep"]["horizons"] = to_int_keys(
            results["recursive_multistep"]["horizons"])
    locked = results.get("locked_test_audit", {})
    for key in ("rows", "observed_target_labels", "local_operational_dates"):
        if key in locked:
            locked[key] = to_int_keys(locked[key])
    difficulty = results.get("configuration_difficulty", {})
    if "persistence_balanced_accuracy" in difficulty:
        difficulty["persistence_balanced_accuracy"] = to_int_keys(
            difficulty["persistence_balanced_accuracy"])
    audit = results.get("data_audit", {})
    # Printed labels are always derived from the stored timestamps, so a change to
    # their formatting takes effect without refitting anything.
    for key, label in (("start", "start_label"), ("end", "end_label")):
        if key in audit:
            audit[label] = f"{pd.Timestamp(audit[key]).day} {pd.Timestamp(audit[key]):%B %Y}"
    return results


if __name__ == "__main__":
    if "--artifacts-only" in sys.argv:
        # Regenerate every table and prose macro from the archived results record.
        archived = json.loads(ARCHIVED_RESULTS.read_text(encoding="utf-8"))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_artifacts(normalize_results(archived), OUTPUT_DIR)
        print(f"Tables and macros written to {OUTPUT_DIR}")
    else:
        analyze(DATA_DIR, OUTPUT_DIR)
