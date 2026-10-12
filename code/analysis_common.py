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
