#!/usr/bin/env python3
"""Forecast metrics, block bootstrap, and transition-event matching."""

from analysis_common import *  # noqa: F401,F403
from models import *  # noqa: F401,F403

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
