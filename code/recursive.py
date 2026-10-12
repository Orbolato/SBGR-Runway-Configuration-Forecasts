#!/usr/bin/env python3
"""Recursive multi-step comparator and prediction summaries."""

from analysis_common import *  # noqa: F401,F403
from models import *  # noqa: F401,F403
from evaluation import *  # noqa: F401,F403

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
