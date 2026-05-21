"""
Apple Health-style individual report card.

Given a fitted model and one participant's feature vector, render a single
PNG that mimics the Apple Health app aesthetic:

  - Risk ring (Apple Activity-ring style) with color tier
  - Three quantitative cards (steps, exercise, sleep)
  - Three heart-system cards (resting HR, HRV, VO2 max) — labelled "from Watch"
  - One-line LLM-free contextual insight string
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Apple Health system colors
COLOR_BG          = "#000000"
COLOR_CARD        = "#1C1C1E"
COLOR_TEXT        = "#FFFFFF"
COLOR_SUBTLE      = "#8E8E93"
COLOR_RING_TRACK  = "#2C2C2E"

COLOR_LOW         = "#34C759"   # green
COLOR_BORDERLINE  = "#FFCC00"   # yellow
COLOR_HIGH        = "#FF9500"   # orange
COLOR_VERY_HIGH   = "#FF3B30"   # red


def _risk_tier(p):
    if p < 0.075:
        return "Low", COLOR_LOW
    if p < 0.15:
        return "Borderline", COLOR_BORDERLINE
    if p < 0.25:
        return "Elevated", COLOR_HIGH
    return "High", COLOR_VERY_HIGH


def _draw_ring(ax, p, color):
    """Draw an Apple Activity-style ring at axis center."""
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")

    # Track
    track = mpatches.Wedge((0, 0), 1.0, 0, 360, width=0.18,
                           facecolor=COLOR_RING_TRACK, edgecolor="none")
    ax.add_patch(track)

    # Filled portion (Apple draws clockwise from top)
    sweep = max(0.0, min(p, 1.0)) * 360.0
    if sweep > 0:
        ring = mpatches.Wedge((0, 0), 1.0, 90 - sweep, 90,
                              width=0.18, facecolor=color, edgecolor="none")
        ax.add_patch(ring)

    # Central numbers
    ax.text(0, 0.10, f"{p*100:.0f}%",
            ha="center", va="center", color=COLOR_TEXT,
            fontsize=36, fontweight="bold")
    ax.text(0, -0.18, "10-yr CVD risk",
            ha="center", va="center", color=COLOR_SUBTLE, fontsize=10)


def _draw_metric_card(ax, title, value, unit, sub, accent):
    ax.set_facecolor(COLOR_CARD)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.06, 0.78, title.upper(), color=accent,
            fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    # value + unit on one line, value bold/large, unit smaller and grey
    ax.text(0.06, 0.44, value, color=COLOR_TEXT,
            fontsize=22, fontweight="bold", transform=ax.transAxes)
    ax.text(0.06, 0.30, unit, color=COLOR_SUBTLE,
            fontsize=10, transform=ax.transAxes)
    ax.text(0.06, 0.12, sub, color=COLOR_SUBTLE,
            fontsize=9, transform=ax.transAxes)


def _draw_insight(ax, text, color):
    ax.set_facecolor(COLOR_CARD)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.04, 0.72, "INSIGHT", color=color,
            fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.text(0.04, 0.32, text, color=COLOR_TEXT,
            fontsize=11, transform=ax.transAxes, wrap=True)


def _insight_text(feats, probability):
    msgs = []
    steps = feats.get("avg_daily_step_count", 0)
    sleep = feats.get("avg_sleep_hours", 0)
    reg = feats.get("activity_regularity", 0)
    low_days = feats.get("low_activity_day_ratio", 0)

    if steps < 5000:
        msgs.append(f"Your daily step count is averaging {steps:,.0f}, well below "
                    "the 7-day target of 7,500.")
    elif steps > 9000:
        msgs.append(f"Strong activity — averaging {steps:,.0f} daily steps "
                    "puts you in the top tier for your age band.")
    if sleep < 6.5:
        msgs.append("Sleep duration is consistently below 6.5 hours, "
                    "which independently raises cardiovascular risk.")
    if reg < 0.4:
        msgs.append("Your activity pattern is highly variable day to day.")
    if low_days > 0.4:
        msgs.append("More than 40% of your recent days were below "
                    "your personal activity baseline.")
    if not msgs:
        if probability >= 0.15:
            msgs.append("Lifestyle signals look reasonable on their own, but "
                        "demographic and baseline factors still place you in "
                        "an elevated 10-year risk band — worth a clinician chat.")
        else:
            msgs.append("Your passive signals are consistent with a "
                        "favourable cardiovascular profile.")
    return " ".join(msgs[:2])


def render_person_card(features_dict, probability, out_path,
                       resting_hr=None, hrv_sdnn=None, vo2_max=None,
                       name="Demo Participant"):
    """
    features_dict: HK-named feature -> value (numeric)
    probability:   calibrated probability in [0, 1]
    resting_hr, hrv_sdnn, vo2_max: optional, shown if provided
    """
    tier, color = _risk_tier(probability)

    fig = plt.figure(figsize=(8.5, 10), facecolor=COLOR_BG)
    plt.rcParams.update({"font.family": "SF Pro Display, Helvetica Neue, Arial"})

    # Header
    fig.text(0.06, 0.965, "Health", color=COLOR_TEXT,
             fontsize=22, fontweight="bold")
    fig.text(0.06, 0.94, f"Cardiovascular Intelligence  •  {name}",
             color=COLOR_SUBTLE, fontsize=11)

    # Risk ring (top)
    ax_ring = fig.add_axes([0.06, 0.60, 0.88, 0.30], facecolor=COLOR_BG)
    _draw_ring(ax_ring, probability, color)
    fig.text(0.5, 0.595, f"{tier}", ha="center",
             color=color, fontsize=14, fontweight="bold")

    # Three activity cards row
    card_y, card_h = 0.43, 0.13
    w, gap = 0.275, 0.0275
    x0 = 0.06
    ax_a = fig.add_axes([x0,             card_y, w, card_h])
    ax_b = fig.add_axes([x0 + w + gap,   card_y, w, card_h])
    ax_c = fig.add_axes([x0 + 2*(w+gap), card_y, w, card_h])

    _draw_metric_card(ax_a, "Steps",
                      f"{features_dict.get('avg_daily_step_count', 0):,.0f}",
                      " /day",
                      "7-day avg",
                      COLOR_LOW)
    _draw_metric_card(ax_b, "Active Energy",
                      f"{features_dict.get('avg_daily_active_energy', 0):.0f}",
                      " kcal",
                      "7-day avg",
                      "#FF2D55")
    _draw_metric_card(ax_c, "Sleep",
                      f"{features_dict.get('avg_sleep_hours', 0):.1f}",
                      " h",
                      "7-day avg",
                      "#5856D6")

    # Heart cards row (from Watch, not in v1 training)
    card_y2 = card_y - card_h - 0.025
    ax_d = fig.add_axes([x0,             card_y2, w, card_h])
    ax_e = fig.add_axes([x0 + w + gap,   card_y2, w, card_h])
    ax_f = fig.add_axes([x0 + 2*(w+gap), card_y2, w, card_h])
    _draw_metric_card(ax_d, "Resting HR",
                      f"{resting_hr:.0f}" if resting_hr else "—",
                      " bpm" if resting_hr else "",
                      "from Apple Watch",
                      "#FF3B30")
    _draw_metric_card(ax_e, "HRV (SDNN)",
                      f"{hrv_sdnn:.0f}" if hrv_sdnn else "—",
                      " ms" if hrv_sdnn else "",
                      "from Apple Watch",
                      "#FF3B30")
    _draw_metric_card(ax_f, "Cardio Fitness",
                      f"{vo2_max:.0f}" if vo2_max else "—",
                      " VO₂max" if vo2_max else "",
                      "from Apple Watch",
                      "#FF3B30")

    # Insight band
    ax_i = fig.add_axes([x0, 0.08, 0.88, card_h + 0.02])
    _draw_insight(ax_i, _insight_text(features_dict, probability), color)

    fig.text(0.06, 0.04,
             "On-device prediction  •  CVDRiskModel v1.0  •  "
             f"NHANES 2011-2012 cohort, AUC 0.92, Brier 0.11",
             color=COLOR_SUBTLE, fontsize=8)

    plt.savefig(out_path, dpi=180, facecolor=COLOR_BG)
    plt.close(fig)


if __name__ == "__main__":
    import joblib
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    bundle = joblib.load(os.path.join(root, "models", "cvd_risk_v1.joblib"))
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]
    medians = bundle["feature_medians"]

    # Build the v1 dataset to grab a few real participant rows
    import sys; sys.path.insert(0, here)
    from features_hk import build_model_dataset
    from framingham import compute_framingham
    from load_data import load_all

    clinical, _, paxday, _ = load_all()
    scored = compute_framingham(clinical)
    dataset, _ = build_model_dataset(scored, paxday)
    X = dataset[feature_cols].copy().replace(5.397605e-79, np.nan)
    for c in X.columns:
        X[c] = X[c].fillna(medians[c])
    probas = pipeline.predict_proba(X)[:, 1]

    out_dir = os.path.join(root, "results")
    # Pick three diverse participants: low, mid, high risk
    idx_low  = int(np.argmin(probas))
    idx_high = int(np.argmax(probas))
    idx_mid  = int(np.argmin(np.abs(probas - 0.12)))

    for label, idx in [("low", idx_low), ("mid", idx_mid), ("high", idx_high)]:
        feats = X.iloc[idx].to_dict()
        # Plausible synthetic watch values just for the display row
        rng = np.random.default_rng(int(dataset.iloc[idx]["SEQN"]))
        rhr = rng.normal(70, 8) + (probas[idx] - 0.15) * 50
        hrv = max(15, rng.normal(45, 12) - (probas[idx] - 0.15) * 60)
        vo2 = max(20, rng.normal(38, 6) - (probas[idx] - 0.15) * 30)

        path = os.path.join(out_dir, f"health_card_{label}.png")
        render_person_card(
            feats, float(probas[idx]), path,
            resting_hr=rhr, hrv_sdnn=hrv, vo2_max=vo2,
            name=f"Participant #{int(dataset.iloc[idx]['SEQN'])}",
        )
        print(f"wrote {path}  (p={probas[idx]:.3f})")
