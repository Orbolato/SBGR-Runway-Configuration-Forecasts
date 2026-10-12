#!/usr/bin/env python3
"""Model families, resampling, temporal selection, and calibration."""

from analysis_common import *  # noqa: F401,F403

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
