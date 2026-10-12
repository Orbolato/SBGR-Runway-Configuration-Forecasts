#!/usr/bin/env python3
"""Draw Figures 1-7 of the R2 manuscript.

Figures 1-3 are study-specific schematics. Figures 4-7 are drawn only from the
serialized results record (``analysis_results.json``) and the plotted series that
it does not hold (``figure_data.json``: monthly configuration dynamics, the wind
profile, the 60-minute reliability curve, and monthly locked-test performance).
No model is refitted here.

Figures 1, 3, and 4 are drawn as in the R1 analysis script. Figures 2, 5, 6, and 7
follow the R2 presentation revision: Figure 2 with widened boxes, Figure 5 with
threshold labels moved off the curve, Figure 6 (former R1 Figure 8) with the
ablation panel as a dot plot on a zoomed axis, and Figure 7 as the standalone
feature-reliance plot (former R1 Figure 7(b)).

Usage:
    python code/make_figures.py [--results results/analysis_results.json]
                                [--figure-data results/figure_data.json]
                                [--out output]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

REPO = Path(__file__).resolve().parents[1]
HORIZONS = ("15", "30", "60")
BLUE, ORANGE, RED = "#1f77b4", "#ff7f0e", "#d62728"
# Fixed metadata keeps the PDF output byte-stable between runs.
PDF_METADATA = {"CreationDate": None}

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)


# --------------------------------------------------------------------- schematics
def draw_fig01(fig_dir: Path):
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    # The current eAIP gives true bearings 73.69/253.67 degrees. The ADC
    # gives the corresponding magnetic bearings 95/275 degrees. Therefore,
    # on a north-up drawing the runway-10 direction rises 16.31 degrees above
    # the horizontal toward the east-northeast.
    angle = np.deg2rad(90.0 - 73.69)
    length, width, x_start = 4.45, 0.30, 0.85
    for y_start, label_left, label_right in [(3.10, "10L", "28R"), (2.15, "10R", "28L")]:
        p0 = np.array([x_start, y_start])
        p1 = p0 + length * np.array([np.cos(angle), np.sin(angle)])
        normal = 0.5 * width * np.array([-np.sin(angle), np.cos(angle)])
        ax.add_patch(Polygon([p0 + normal, p1 + normal, p1 - normal, p0 - normal],
                             closed=True, color="#606060"))
        ax.text(p0[0] - 0.10, p0[1] + 0.03, label_left, weight="bold",
                ha="right", va="center")
        ax.text(p1[0] + 0.08, p1[1] - 0.02, label_right, weight="bold",
                ha="left", va="center")
    ax.annotate("", xy=(0.38, 1.55), xytext=(0.38, 0.70),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
    ax.text(0.38, 1.62, "True N", ha="center", va="bottom", fontsize=8)
    ax.text(3.0, 4.72, "SBGR reciprocal runway directions (north-up schematic)",
            ha="center", weight="bold")
    ax.text(3.0, 1.45, "10 configuration: arrivals 10R | departures 10L\n"
                       "28 configuration: arrivals 28L | departures 28R",
             ha="center", va="top")
    ax.text(3.0, 0.60, "True bearings 74°/254° (used for wind components)",
            ha="center", va="center", fontsize=8)
    ax.text(3.0, 0.24, "Magnetic 95°/275°", ha="center", va="center", fontsize=8)
    x0, y0 = 6.3, 3.6
    ax.plot([x0, 11.4], [y0, y0], color="black", lw=1)
    ax.scatter([x0, x0 + .9, x0 + 1.8, x0 + 3.6], [y0] * 4, color="black", s=18, zorder=3)
    ax.text(x0, y0 + .35, r"issue $t$", ha="center")
    for dx, lab in [(0.9, "+15"), (1.8, "+30"), (3.6, "+60 min")]:
        ax.text(x0 + dx, y0 + .35, lab, ha="center")
    ax.text(8.85, 2.65, "Available or assumed available at issue time", ha="center", weight="bold")
    ax.text(8.85, 2.2, "last completed configuration and movements\n"
                        "latest METAR; published schedule for target bin",
            ha="center", va="top")
    ax.text(8.85, 1.05, "Target", ha="center", weight="bold")
    ax.text(8.85, .65, "configuration observed at each horizon", ha="center")
    fig.tight_layout()
    _save(fig, fig_dir / "Figure_01.pdf")


def draw_fig02(fig_dir: Path):
    """Analytical workflow (R2 version: the third and fourth boxes are widened)."""
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    boxes = [
        (0.1, 3.9, 2.1, 1.2, "Operational records\nSchedules | METAR"),
        (2.45, 3.9, 2.1, 1.2, "15-min integration\nquality control"),
        (4.8, 3.9, 2.4, 1.2, "Features and targets\nunder availability rules"),
        (7.45, 3.9, 2.45, 1.2, "2022 initial fit\n2023 rolling validation"),
        (10.15, 3.9, 1.65, 1.2, "Locked 2024\ntest"),
        (3.2, 1.3, 2.4, 1.2, "Baselines and\nrepresentative models"),
        (6.1, 1.3, 2.4, 1.2, "Block-bootstrap\npaired comparisons"),
        (9.0, 1.3, 2.8, 1.2, "Transitions | calibration\nrobustness | explanations"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                    facecolor="#edf2f7", edgecolor="#334e68"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")
    arrows = [((2.2, 4.5), (2.45, 4.5)), ((4.55, 4.5), (4.8, 4.5)),
              ((7.2, 4.5), (7.45, 4.5)), ((9.9, 4.5), (10.15, 4.5)),
              ((6.0, 3.9), (4.4, 2.5)), ((5.6, 1.9), (6.1, 1.9)),
              ((8.5, 1.9), (9.0, 1.9))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=10,
                                     color="#334e68", lw=1))
    fig.tight_layout()
    _save(fig, fig_dir / "Figure_02.pdf")


def draw_fig03(fig_dir: Path):
    """Concept of operations: feeds, forecast production, outputs, and their use.

    The canvas is 100 units wide and the drawing is emitted close to its printed
    width, so label sizes on the page match the sizes set here.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ax.set_xlim(-15, 101); ax.set_ylim(0, 92); ax.axis("off")
    body, header, band_label = 5.2, 6.0, 5.8

    bands = [
        (77.0, 13.0, "Inputs at $t$", "#eef3f8"),
        (55.0, 16.0, "Production", "#f3f0ea"),
        (28.5, 20.5, "Advisory output", "#eef3f8"),
        (4.0, 20.0, "Operational use", "#f3f0ea"),
    ]
    for y, h, title, color in bands:
        ax.add_patch(FancyBboxPatch((0, y), 100, h, boxstyle="round,pad=0.4",
                                    facecolor=color, edgecolor="#c3ccd6", lw=0.8))
        ax.text(-8.0, y + h / 2, title, ha="center", va="center", rotation=90,
                fontsize=band_label, weight="bold", color="#334e68")

    def box(x, y, w, h, edge, lines, title=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    facecolor="white", edgecolor=edge, lw=0.9))
        if title:
            ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top",
                    fontsize=header, weight="bold")
            ax.text(x + w / 2, y + (h - 6.0) / 2, "\n".join(lines), ha="center",
                    va="center", fontsize=body, linespacing=1.45)
        else:
            ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center",
                    va="center", fontsize=body, linespacing=1.45)

    quarters = [2.0, 26.5, 51.0, 75.5]
    feeds = [
        ["Tower movements", "and runway use"],
        ["Published schedule", "for the target bin"],
        ["Latest METAR,", "age at most 120 min"],
        ["Calendar and", "local-time terms"],
    ]
    for x, lines in zip(quarters, feeds):
        box(x, 79.0, 22.5, 9.0, "#334e68", lines)

    thirds = [2.0, 35.0, 68.0]
    steps = [
        ["Features under availability rules:", "33 predictors,", "six groups"],
        ["Horizon models fitted", "on 2022–2023,", "locked 2024 test"],
        ["Probability calibration", "and validation-", "selected threshold"],
    ]
    for x, lines in zip(thirds, steps):
        box(x, 57.0, 30.0, 12.0, "#8a6d3b", lines)

    outputs = [
        ("+15 min", ["Logistic regression", "P(28) and P(10)"]),
        ("+30 min", ["Gradient boosting", "P(28) and P(10)"]),
        ("+60 min", ["Gradient boosting", "P(28) and P(10)"]),
    ]
    for x, (title, lines) in zip(thirds, outputs):
        box(x, 32.5, 30.0, 14.0, "#334e68", lines, title=title)
    ax.text(50, 30.3, "Shown with the last confirmed configuration, the wind cue, "
                      "and the age of every input",
            ha="center", va="center", fontsize=body, style="italic", color="#334e68")

    uses = [
        ["Tower and ground", "movement control:", "taxi routing and", "holding-point readiness"],
        ["Approach sequencing", "and departure-queue", "preparation; briefing", "of the next state"],
        ["Flow management and", "airport operations center:", "arrival rate, stand", "and gate planning"],
    ]
    for x, lines in zip(thirds, uses):
        box(x, 6.0, 30.0, 16.0, "#8a6d3b", lines)
    ax.text(50, 0.2, "Authorized operational personnel retain the configuration decision; "
                     "forecasts are advisory\nand are logged with operator acknowledgement",
            ha="center", va="center", fontsize=body, weight="bold",
            color="#334e68", linespacing=1.45)

    for y0, y1 in ((77.0, 71.0), (55.0, 49.5), (28.5, 24.0)):
        for x in (17.0, 50.0, 83.0):
            ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                         mutation_scale=8, color="#334e68", lw=0.9))
    fig.tight_layout()
    _save(fig, fig_dir / "Figure_03.pdf")


# --------------------------------------------------------------------- analytical
def draw_fig04(fig_dir: Path, figure_data: dict):
    """Configuration dynamics and runway-relative wind over the full record."""
    dyn = figure_data["configuration_dynamics"]
    wind = figure_data["wind_profile"]
    p28 = np.asarray(dyn["p28"])
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8))
    x = np.arange(len(p28))
    axes[0].plot(x, p28, color="#1f77b4", marker="o", ms=2, lw=1)
    axes[0].set_ylabel("Share of 28 configuration")
    axes[0].set_ylim(0, max(.45, p28.max() * 1.1))
    axes[1].plot(x, 100 * np.asarray(dyn["transition_rate"]), color="#d95f02", marker="s", ms=2, lw=1)
    axes[1].set_ylabel("State changes (% of intervals)")
    axes[2].plot(np.arange(len(wind["share_28"])), wind["share_28"], color="#333333", marker="D", ms=3)
    axes[2].set_xticks(np.arange(len(wind["share_28"])))
    axes[2].set_xticklabels(wind["bins"], rotation=45, ha="right")
    axes[2].set_ylabel("Share of 28 configuration")
    axes[2].set_xlabel("Advantage for direction 28 (kt)")
    labels = list(dyn["month"])
    ticks = np.arange(0, len(labels), 3)
    for ax in axes[:2]:
        ax.set_xticks(ticks); ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right")
        ax.grid(axis="y", color="#dddddd", lw=.5)
    axes[2].grid(axis="y", color="#dddddd", lw=.5)
    axes[0].set_title("(a) Configuration prevalence")
    axes[1].set_title("(b) Transition frequency")
    axes[2].set_title("(c) Runway-relative wind")
    fig.tight_layout()
    _save(fig, fig_dir / "Figure_04.pdf")


def draw_fig05(fig_dir: Path, results: dict, figure_data: dict):
    """60-minute alert-threshold trade-off and reliability of the selected model."""
    tradeoff = results["threshold_tradeoff_60min"]
    curve = np.array([(t["false_alerts_per_day"], t["event_recall"]) for t in tradeoff])
    rel = figure_data["reliability_60min_selected"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    ax = axes[0]
    ax.plot(curve[:, 0], curve[:, 1], color=BLUE, marker="o", ms=3)
    # Offsets (points) chosen so that no label sits on the curve.
    offsets = {0.10: (-4, 4, "right"), 0.20: (4, -9, "left"), 0.30: (-4, 4, "right"),
               0.40: (4, -2, "left"), 0.50: (0, 5, "center"), 0.60: (-5, -3, "right"),
               0.70: (-5, -6, "right"), 0.80: (0, 5, "center"), 0.90: (0, 5, "center")}
    for t in tradeoff[::2]:
        dx, dy, ha = offsets[round(t["threshold"], 2)]
        ax.annotate(f"{t['threshold']:.2f}", (t["false_alerts_per_day"], t["event_recall"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=6, ha=ha)
    m60 = results["models"]["60"]
    selected = m60[m60["selected_learned_model"]]["selection_threshold"]
    chosen = min(tradeoff, key=lambda t: abs(t["threshold"] - selected))
    ax.scatter([chosen["false_alerts_per_day"]], [chosen["event_recall"]], s=45,
               facecolors="none", edgecolors=RED, lw=1.2, zorder=3)
    ax.annotate("nearest to validation-\nselected threshold",
                (chosen["false_alerts_per_day"], chosen["event_recall"]),
                xytext=(2.45, 0.335), textcoords="data", fontsize=6.5, color=RED, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8, shrinkA=1, shrinkB=4))
    ax.set_xlim(0.7, 5.9)
    ax.set_xlabel("False transition alerts per day")
    ax.set_ylabel(r"Event recall (target within $\pm$15 min)")
    ax.set_title("(a) 60-min alert-threshold trade-off")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color="#777777", ls="--", lw=1)
    ax.plot(rel["mean_predicted"], rel["observed"], color=BLUE, marker="o", ms=3)
    ax.set_xlabel("Predicted probability of 28 configuration")
    ax.set_ylabel("Observed frequency")
    ax.set_title("(b) Reliability, selected 60-min model")
    fig.tight_layout()
    _save(fig, fig_dir / "Figure_05.pdf")


def draw_fig06(fig_dir: Path, results: dict, figure_data: dict, model_labels: dict):
    """Monthly robustness and feature-group ablation for the 60-minute forecast."""
    monthly = [m for m in figure_data["monthly_performance"] if str(m["horizon"]) == "60"]
    months = [m["month"] for m in monthly]
    selected_label = model_labels[results["models"]["60"]["selected_learned_model"]]
    ablation = results["ablation_60min"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
    ax = axes[0]
    ax.plot(months, [m["selected"] for m in monthly], marker="o", label=selected_label)
    ax.plot(months, [m["persistence"] for m in monthly], marker="s", ls="--", label="Persistence")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("(a) Monthly robustness, 60 min")
    ax.legend(frameon=False)
    ax = axes[1]
    labels = list(ablation)
    values = [ablation[k]["balanced_accuracy"] for k in labels]
    pos = np.arange(len(labels))
    ax.hlines(pos, 0.86, values, color="#c9d6e3", lw=1)
    ax.plot(values, pos, "o", color="#4c78a8", ms=5)
    for p, v in zip(pos, values):
        ax.annotate(f"{v:.3f}", (v, p), xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.5)
    ax.set_yticks(pos, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.86, 0.955)
    ax.set_xlabel("Balanced accuracy")
    ax.grid(axis="x", color="#dddddd", lw=.5)
    ax.set_axisbelow(True)
    ax.set_title("(b) Feature-group ablation, 60 min")
    fig.tight_layout()
    _save(fig, fig_dir / "Figure_06.pdf")


def draw_fig07(fig_dir: Path, results: dict, feature_labels: dict):
    """Permutation importance of gradient boosting at all three horizons."""
    importance = results["permutation_importance_boosting"]
    lookup = {h: {e["feature"]: e["mean"] for e in importance[h]} for h in HORIZONS}
    candidates = {e["feature"] for h in HORIZONS for e in importance[h][:6]}
    ranked = sorted(candidates, key=lambda f: (-max(lookup[h][f] for h in HORIZONS), f))
    with plt.rc_context({"font.size": 9, "axes.titlesize": "large", "axes.labelsize": "medium",
                         "legend.fontsize": "medium"}):
        fig, ax = plt.subplots(figsize=(7.1, 3.2))
        positions = np.arange(len(ranked))
        for offset, h, color in zip((-.26, 0, .26), HORIZONS, ("#1f77b4", "#d95f02", "#7570b3")):
            ax.barh(positions + offset, [lookup[h][f] for f in ranked], height=.26, color=color,
                    label=f"{h} min")
        ax.set_yticks(positions, [feature_labels[f] for f in ranked])
        ax.invert_yaxis()
        ax.set_xlabel("Loss in balanced accuracy when permuted")
        ax.grid(axis="x", color="#dddddd", lw=.5)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        _save(fig, fig_dir / "Figure_07.pdf")


def draw_all(results: dict, figure_data: dict, out: Path,
             feature_labels: dict | None = None, model_labels: dict | None = None) -> None:
    """Draw Figures 1-7. ``results`` uses the JSON (string-keyed) layout."""
    if feature_labels is None or model_labels is None:
        from run_analysis import FEATURE_DEFINITIONS, MODEL_LABELS
        feature_labels = {k: v[0] for k, v in FEATURE_DEFINITIONS.items()}
        model_labels = dict(MODEL_LABELS)
    out.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        draw_fig01(out)
        draw_fig02(out)
        draw_fig03(out)
        draw_fig04(out, figure_data)
        draw_fig05(out, results, figure_data)
        draw_fig06(out, results, figure_data, model_labels)
        draw_fig07(out, results, feature_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=REPO / "results" / "analysis_results.json")
    parser.add_argument("--figure-data", type=Path, default=REPO / "results" / "figure_data.json")
    parser.add_argument("--out", type=Path, default=REPO / "output")
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    figure_data = json.loads(args.figure_data.read_text(encoding="utf-8"))
    draw_all(results, figure_data, args.out)
    print(f"Figures 1-7 written to {args.out}")


if __name__ == "__main__":
    main()
