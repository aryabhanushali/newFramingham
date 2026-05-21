"""
Adaptive Cardiovascular Health Intelligence — interactive demo.

A web companion to the Python/Core ML pipeline in this repo: it loads the same
calibrated scikit-learn model the iOS app uses (`models/cvd_risk_v1.joblib`)
and predicts 10-year cardiovascular risk from HealthKit-shaped passive signals
(activity, sleep, light). No data leaves the browser session.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Page chrome — Apple Health-ish aesthetic.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive CVD Risk",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APPLE_RED   = "#FF3B30"
APPLE_AMBER = "#FF9500"
APPLE_GREEN = "#34C759"
APPLE_GRAY  = "#8E8E93"
APPLE_BLUE  = "#0A84FF"
PANEL_BG    = "#F2F2F7"
INK         = "#1C1C1E"

st.markdown(
    f"""
    <style>
      .main .block-container {{ padding-top: 2rem; max-width: 1180px; }}
      h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
      .risk-hero {{
        background: {PANEL_BG}; border-radius: 22px; padding: 28px 32px;
        margin-bottom: 18px;
      }}
      .risk-hero .big {{
        font-size: 76px; font-weight: 700; line-height: 1; letter-spacing: -0.03em;
      }}
      .risk-hero .label {{
        font-size: 13px; color: {APPLE_GRAY}; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 6px;
      }}
      .pill {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: 13px; font-weight: 600;
      }}
      .feature-cap {{ color: {APPLE_GRAY}; font-size: 11px; margin-top: -8px; }}
      .footnote {{ color: {APPLE_GRAY}; font-size: 12px; line-height: 1.5; }}
      [data-testid="stSidebar"] {{ background: #FAFAFC; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "cvd_risk_v1.joblib")
META_PATH  = os.path.join(HERE, "models", "training_metadata.json")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["feature_cols"], bundle["feature_medians"]


@st.cache_resource
def load_metadata():
    with open(META_PATH) as fh:
        return json.load(fh)


pipeline, FEATURE_COLS, FEATURE_MEDIANS = load_model()
META = load_metadata()


# ---------------------------------------------------------------------------
# Feature catalogue — drives the sidebar UI. Mirrors src/healthkit_schema.py
# so each Streamlit control is labelled with its HealthKit identifier and the
# clinician-readable description.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Input:
    key: str
    label: str
    hk_id: str
    unit: str
    min: float
    max: float
    step: float
    description: str
    group: str


INPUTS: list[Input] = [
    Input("age",                "Age",                          "HKCharacteristicTypeIdentifierDateOfBirth",
          "years",  30, 80, 1.0,
          "Age in years.", "Demographics"),
    Input("biological_sex",     "Biological sex",               "HKCharacteristicTypeIdentifierBiologicalSex",
          "enum",   0, 1, 1.0,
          "0 = female, 1 = male (binary encoding required by the model).", "Demographics"),

    Input("avg_daily_step_count",     "Average daily steps",     "HKQuantityTypeIdentifierStepCount",
          "steps",  500, 25000, 100.0,
          "Mean steps/day over the past 7 days.", "Activity"),
    Input("avg_daily_active_energy",  "Average active energy",   "HKQuantityTypeIdentifierActiveEnergyBurned",
          "kcal",   50, 1500, 10.0,
          "Mean kcal/day burned in active movement.", "Activity"),
    Input("avg_daily_exercise_min",   "Exercise minutes",        "HKQuantityTypeIdentifierAppleExerciseTime",
          "min",    0, 180, 1.0,
          "Mean minutes/day above brisk-walking intensity.", "Activity"),
    Input("peak_intensity",           "Peak intensity (kcal)",   "HKQuantityTypeIdentifierActiveEnergyBurned",
          "kcal",   50, 2500, 10.0,
          "Peak single-day active energy.", "Activity"),
    Input("activity_regularity",      "Activity regularity",     "derived",
          "0–1",    0.1, 1.0, 0.01,
          "1 / (1 + std of daily activity). Higher = more consistent day-to-day.", "Activity"),
    Input("low_activity_day_ratio",   "Low-activity day ratio",  "derived",
          "0–1",    0.0, 1.0, 0.01,
          "Fraction of recorded days below this person's 25th percentile.", "Activity"),
    Input("sedentary_minutes",        "Sedentary minutes",       "HKQuantityTypeIdentifierAppleStandTime",
          "min",    100, 1300, 5.0,
          "Mean sedentary minutes/day (inverse of stand-time).", "Activity"),

    Input("avg_sleep_hours",          "Average sleep",           "HKCategoryTypeIdentifierSleepAnalysis",
          "hours",  3.0, 12.0, 0.1,
          "Mean nightly sleep duration.", "Sleep"),
    Input("sleep_regularity",         "Sleep regularity",        "derived",
          "0–1",    0.05, 1.0, 0.01,
          "1 / (1 + std of nightly sleep hours).", "Sleep"),

    Input("circadian_light_exposure", "Daytime light exposure",  "derived (ambient light)",
          "lux",    1000, 600000, 1000.0,
          "Mean ambient light during waking hours.", "Circadian"),
]

INPUT_BY_KEY = {inp.key: inp for inp in INPUTS}


# ---------------------------------------------------------------------------
# Sidebar: data entry. Defaults seeded from training-set medians.
# All widget state lives in session_state under the feature key; the Reset
# button simply rewrites those entries and triggers a rerun.
# ---------------------------------------------------------------------------
SEX_LABELS = ["Female", "Male"]


def _initial_value(key):
    """Convert a feature median into the widget's native type."""
    med = FEATURE_MEDIANS[key]
    if key == "biological_sex":
        return SEX_LABELS[int(med)]
    if key in {"age", "avg_daily_step_count", "avg_daily_active_energy",
               "avg_daily_exercise_min", "peak_intensity",
               "sedentary_minutes", "circadian_light_exposure"}:
        return int(round(med))
    return float(med)


for inp in INPUTS:
    if inp.key not in st.session_state:
        st.session_state[inp.key] = _initial_value(inp.key)


st.sidebar.markdown("### Inputs")
st.sidebar.caption(
    "Each control maps to a HealthKit identifier the iOS counterpart reads "
    "from an Apple Watch. Defaults are the NHANES-2011-2012 training-set median "
    "for each signal."
)

if st.sidebar.button("Reset to cohort median", width="stretch"):
    for inp in INPUTS:
        st.session_state[inp.key] = _initial_value(inp.key)
    st.rerun()

with st.sidebar.expander("Demographics", expanded=True):
    st.slider("Age (years)", 30, 80, key="age")
    st.radio("Biological sex", SEX_LABELS, horizontal=True, key="biological_sex")

with st.sidebar.expander("Activity (Apple Watch)", expanded=True):
    st.slider("Daily steps", 500, 25000, step=100, key="avg_daily_step_count",
              help="HKQuantityTypeIdentifierStepCount, mean over last 7 days")
    st.slider("Active energy (kcal/day)", 50, 1500, step=10,
              key="avg_daily_active_energy",
              help="HKQuantityTypeIdentifierActiveEnergyBurned")
    st.slider("Exercise minutes / day", 0, 180,
              key="avg_daily_exercise_min",
              help="HKQuantityTypeIdentifierAppleExerciseTime")
    st.slider("Peak day intensity (kcal)", 50, 2500, step=10,
              key="peak_intensity")
    st.slider("Activity regularity", 0.10, 1.00, step=0.01,
              key="activity_regularity",
              help="Higher = more consistent day-to-day movement")
    st.slider("Low-activity day fraction", 0.00, 1.00, step=0.01,
              key="low_activity_day_ratio")
    st.slider("Sedentary minutes / day", 100, 1300, step=5,
              key="sedentary_minutes",
              help="HKQuantityTypeIdentifierAppleStandTime (inverse)")

with st.sidebar.expander("Sleep", expanded=True):
    st.slider("Average sleep (hours)", 3.0, 12.0, step=0.1,
              key="avg_sleep_hours",
              help="HKCategoryTypeIdentifierSleepAnalysis")
    st.slider("Sleep regularity", 0.05, 1.00, step=0.01,
              key="sleep_regularity",
              help="Higher = more consistent nightly sleep duration")

with st.sidebar.expander("Circadian", expanded=False):
    st.slider("Daytime light exposure (lux)", 1000, 600000, step=1000,
              key="circadian_light_exposure")


# Assemble the feature vector in the exact training order.
features = {col: float(st.session_state[col])
            for col in FEATURE_COLS if col != "biological_sex"}
features["biological_sex"] = float(SEX_LABELS.index(st.session_state["biological_sex"]))
X = pd.DataFrame([[features[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
risk = float(pipeline.predict_proba(X)[0, 1])


# ---------------------------------------------------------------------------
# Risk categorisation — Framingham/AHA convention.
# ---------------------------------------------------------------------------
def categorise(p: float):
    if p < 0.05:
        return "Low", APPLE_GREEN, "Below 5%. Continue current habits."
    if p < 0.10:
        return "Borderline", APPLE_AMBER, "Between 5% and 10%. Watch trends over time."
    if p < 0.20:
        return "Intermediate", APPLE_AMBER, "Between 10% and 20%. Lifestyle changes can shift this materially."
    return "High", APPLE_RED, "Above 20%. Worth discussing with a clinician."


category, color, interp = categorise(risk)


# ---------------------------------------------------------------------------
# Hero: title + big number.
# ---------------------------------------------------------------------------
st.title("Adaptive Cardiovascular Health Intelligence")
st.markdown(
    "**A calibrated 10-year cardiovascular risk estimate built from passive "
    "Apple Watch-style signals — no blood draws, no clinic visit.** Trained on "
    f"{META['training_n']:,} adults from NHANES 2011-2012 against Framingham "
    f"labels. Cross-validated AUC **{META['metrics']['auc_cv5']:.2f}**, Brier "
    f"**{META['metrics']['brier_cv5']:.2f}** after isotonic calibration. The "
    "same model artifact powers the bundled Core ML iOS prototype."
)

hero_left, hero_right = st.columns([1.2, 1])

with hero_left:
    st.markdown(
        f"""
        <div class="risk-hero">
          <div class="label">Estimated 10-year CVD risk</div>
          <div class="big" style="color:{color};">{risk*100:.1f}%</div>
          <div style="margin-top:10px;">
            <span class="pill" style="background:{color}22; color:{color};">{category}</span>
            <span style="color:{APPLE_GRAY}; margin-left:10px;">{interp}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    # Calibrated risk gauge — visual on top of the number.
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        number={"suffix": "%", "font": {"size": 36, "color": INK}},
        gauge={
            "axis": {"range": [0, 50], "tickwidth": 1, "tickcolor": APPLE_GRAY},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 5],   "color": "#E8F8EE"},
                {"range": [5, 10],  "color": "#FFF4E0"},
                {"range": [10, 20], "color": "#FFE9D6"},
                {"range": [20, 50], "color": "#FFE0DE"},
            ],
            "threshold": {"line": {"color": APPLE_GRAY, "width": 2}, "thickness": 0.85, "value": 10},
        },
    ))
    gauge.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor=PANEL_BG, font={"family": "-apple-system, system-ui"})
    st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Interpretability: local sensitivity. Vary each feature ±1 cohort SD and see
# how much the prediction moves. Cheap, model-agnostic, and the resulting bar
# chart is the most useful "why this number?" panel for a non-technical user.
# ---------------------------------------------------------------------------
@st.cache_data
def cohort_spreads():
    """Rough ±range each feature can plausibly move, used for sensitivity."""
    return {
        "age":                       (30,    80,   5.0),
        "biological_sex":            (0,     1,    1.0),
        "avg_daily_active_energy":   (50,    1500, 120.0),
        "avg_daily_step_count":      (500,   25000, 2500.0),
        "avg_daily_exercise_min":    (0,     180,  15.0),
        "activity_regularity":       (0.1,   1.0,  0.10),
        "low_activity_day_ratio":    (0.0,   1.0,  0.10),
        "peak_intensity":            (50,    2500, 200.0),
        "sedentary_minutes":         (100,   1300, 120.0),
        "avg_sleep_hours":           (3.0,   12.0, 1.0),
        "sleep_regularity":          (0.05,  1.0,  0.10),
        "circadian_light_exposure":  (1000,  600000, 50000.0),
    }


def local_sensitivity(features: dict) -> pd.DataFrame:
    base_vec = pd.DataFrame([[features[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
    base = float(pipeline.predict_proba(base_vec)[0, 1])
    spreads = cohort_spreads()
    rows = []
    for col in FEATURE_COLS:
        lo, hi, step = spreads[col]
        v = features[col]
        up_v = min(hi, v + step)
        dn_v = max(lo, v - step)
        up = base_vec.copy(); up[col] = up_v
        dn = base_vec.copy(); dn[col] = dn_v
        p_up = float(pipeline.predict_proba(up)[0, 1])
        p_dn = float(pipeline.predict_proba(dn)[0, 1])
        rows.append({
            "feature": col,
            "delta_up":   p_up - base,
            "delta_down": p_dn - base,
            "swing": abs(p_up - base) + abs(p_dn - base),
        })
    return pd.DataFrame(rows).sort_values("swing", ascending=False)


sens = local_sensitivity(features)


st.markdown("---")
left, right = st.columns([1.1, 1])

with left:
    st.markdown("#### What's moving your number?")
    st.caption(
        "Each bar shows how much your estimated risk would change if that single "
        "input moved by one cohort step in either direction, with everything "
        "else held fixed. The longer the bar, the more leverage that signal has."
    )

    top = sens.head(6).iloc[::-1]
    bar = go.Figure()
    bar.add_trace(go.Bar(
        y=top["feature"], x=top["delta_down"] * 100,
        orientation="h", name="If lower",
        marker_color=APPLE_GREEN, hovertemplate="%{x:+.1f} pp<extra>If lower</extra>",
    ))
    bar.add_trace(go.Bar(
        y=top["feature"], x=top["delta_up"] * 100,
        orientation="h", name="If higher",
        marker_color=APPLE_RED, hovertemplate="%{x:+.1f} pp<extra>If higher</extra>",
    ))
    bar.update_layout(
        barmode="relative", height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Change in predicted risk (percentage points)",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white",
        font={"family": "-apple-system, system-ui"},
    )
    bar.add_vline(x=0, line_color=APPLE_GRAY, line_width=1)
    st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})

with right:
    st.markdown("#### Where you stand")
    st.caption(
        "Your selected inputs sit at these percentiles of the 12-feature "
        "vector relative to the training cohort median (defaults are the "
        "cohort median itself)."
    )
    base_rows = []
    for col in FEATURE_COLS:
        med = FEATURE_MEDIANS[col]
        cur = features[col]
        if med == 0:
            delta = 0.0
        else:
            delta = (cur - med) / abs(med) * 100
        base_rows.append({"feature": col, "you": cur, "median": med, "delta_pct": delta})
    pct_df = pd.DataFrame(base_rows)
    st.dataframe(
        pct_df.style.format({"you": "{:,.1f}", "median": "{:,.1f}", "delta_pct": "{:+.0f}%"})
                    .background_gradient(subset=["delta_pct"], cmap="RdYlGn_r", vmin=-100, vmax=100),
        hide_index=True, width="stretch", height=320,
    )


# ---------------------------------------------------------------------------
# What-if explorer — pick one lever and watch the risk move.
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### What-if: change one thing")
st.caption(
    "Pick a single signal and sweep it across its plausible range while the "
    "other eleven inputs stay at your current values. This is the panel a "
    "user would interact with after seeing their number — it answers "
    "“what would actually move this?”"
)

knob_col, plot_col = st.columns([1, 2.2])
with knob_col:
    knob = st.selectbox(
        "Signal to sweep",
        options=[c for c in FEATURE_COLS if c not in ("biological_sex",)],
        format_func=lambda c: INPUT_BY_KEY[c].label,
    )

lo, hi, _ = cohort_spreads()[knob]
sweep_values = np.linspace(lo, hi, 60)
sweep_rows = []
for v in sweep_values:
    row = features.copy()
    row[knob] = float(v)
    Xv = pd.DataFrame([[row[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
    sweep_rows.append({"x": float(v), "risk": float(pipeline.predict_proba(Xv)[0, 1]) * 100})
sweep_df = pd.DataFrame(sweep_rows)

with plot_col:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep_df["x"], y=sweep_df["risk"], mode="lines",
        line=dict(color=APPLE_BLUE, width=3),
        hovertemplate=f"{INPUT_BY_KEY[knob].label}: %{{x:,.1f}} "
                      f"{INPUT_BY_KEY[knob].unit}<br>Risk: %{{y:.1f}}%<extra></extra>",
    ))
    fig.add_hline(y=10, line_dash="dash", line_color=APPLE_GRAY,
                  annotation_text="10% clinical threshold", annotation_position="top left")
    fig.add_vline(x=features[knob], line_color=APPLE_RED,
                  annotation_text="You", annotation_position="top right")
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=f"{INPUT_BY_KEY[knob].label} ({INPUT_BY_KEY[knob].unit})",
        yaxis_title="Predicted 10-yr CVD risk (%)",
        plot_bgcolor="white", font={"family": "-apple-system, system-ui"},
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Methodology + honest limitations footer.
# ---------------------------------------------------------------------------
st.markdown("---")
mcol1, mcol2, mcol3 = st.columns(3)

with mcol1:
    st.markdown("##### Model")
    st.markdown(
        f"- **Framework:** {META['framework']}\n"
        f"- **Calibration:** {META['calibration']}\n"
        f"- **Cohort:** {META['training_cohort']}\n"
        f"- **Training N:** {META['training_n']:,}"
    )

with mcol2:
    st.markdown("##### Cross-validation")
    st.markdown(
        f"- **AUC (5-fold):** {META['metrics']['auc_cv5']:.3f}\n"
        f"- **Brier (5-fold):** {META['metrics']['brier_cv5']:.3f}\n"
        f"- **High-risk threshold:** {META['high_risk_threshold']*100:.0f}%\n"
        f"- **High-risk prevalence:** {META['metrics']['high_risk_prevalence']*100:.1f}%"
    )

with mcol3:
    st.markdown("##### On-device parity")
    st.markdown(
        "- Same feature vector as the bundled `CVDRiskModel.mlmodel`\n"
        "- Schema in `src/healthkit_schema.py`\n"
        "- iOS prototype reads the same HK identifiers shown next to each "
        "sidebar control\n"
        "- Core ML inference is sub-millisecond on A14+"
    )

st.markdown(
    f"""
    <div class="footnote" style="margin-top:18px;">
      <b>Honest limitations.</b> NHANES 2011-2012 is a US cross-sectional sample;
      the Framingham coefficients themselves come from a non-representative cohort.
      Activity/energy units are back-fit from raw accelerometer counts to match
      CDC adult medians, which makes population trends honest but individual
      conversions noisy. The score is a calibrated probability of being labelled
      high-risk by Framingham — one degree removed from actual CVD events. Use
      this for self-awareness and triage; clinical decisions belong with a
      physician. Heart-rate / HRV / VO₂max are read on-device in the iOS app
      but not part of the v1 training set because NHANES 2011-2012 doesn't
      include them.
      <br><br>
      Data: NHANES (CDC). Labels: D'Agostino R.B. et al., 2008, "General
      Cardiovascular Risk Profile for Use in Primary Care," <i>Circulation</i>
      117(6).
    </div>
    """,
    unsafe_allow_html=True,
)
