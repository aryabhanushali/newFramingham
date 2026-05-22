"""
A New Framingham — interactive risk estimate from passive Apple Watch signals.

Web companion to the Python/Core ML pipeline in this repo. Loads the same
calibrated scikit-learn bundle the iOS app uses (`models/cvd_risk_v1.joblib`)
and predicts 10-year cardiovascular disease risk from activity, sleep, and
demographics — no cholesterol panel, no clinic visit. Nothing leaves the
browser session.

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
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="A New Framingham",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/aryabhanushali/newframingham",
        "Report a bug": "https://github.com/aryabhanushali/newframingham/issues",
        "About": (
            "A New Framingham — a re-creation of the 1948 Framingham Heart "
            "Study's 10-year cardiovascular risk model, fed by passive Apple "
            "Watch–style signals instead of bloodwork.\n\n"
            "Built by Arya Bhanushali · github.com/aryabhanushali/newframingham"
        ),
    },
)

# ---------------------------------------------------------------------------
# Visual language — editorial: serif headlines, refined accent palette
# ---------------------------------------------------------------------------
INK         = "#15161A"       # primary text
INK_SOFT    = "#5C6066"       # secondary text
RULE        = "#E3E2DC"       # hairline rules
PAPER       = "#FAF8F2"       # warm off-white panel background
CRIMSON     = "#B3261E"       # editorial red — high risk, accents
OCHRE       = "#B98123"       # mustard — intermediate risk
SAGE        = "#3F6E4A"       # sage — low risk
CHARCOAL    = "#2A2C32"       # near-black accent

st.markdown(
    f"""
    <style>
      /* Page chrome */
      .main .block-container {{
        padding-top: 1.6rem; padding-bottom: 4rem;
        max-width: 1140px;
      }}
      [data-testid="stSidebar"] {{
        background: #FBF9F3; border-right: 1px solid {RULE};
      }}

      /* Force ink-on-paper for every label and body element so the page
         reads correctly regardless of the visitor's browser/system theme. */
      html, body, .stApp, .stMarkdown, p, span, label, div {{
        color: {INK};
      }}
      [data-testid="stSidebar"] *,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] span,
      .stSlider label, .stRadio label, .stSelectbox label,
      .stSlider [data-baseweb], .stRadio [data-baseweb], .stSelectbox [data-baseweb],
      [data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"] p {{
        color: {INK} !important;
      }}
      /* Streamlit's expander chevron + header */
      [data-testid="stExpander"] summary,
      [data-testid="stExpander"] summary p,
      [data-testid="stExpander"] details {{
        color: {INK} !important;
      }}
      /* Help-tooltip "?" icon should be dim, not invisible */
      [data-testid="stTooltipIcon"] svg {{ fill: {INK_SOFT}; }}

      /* Editorial typography */
      h1, h2 {{
        font-family: "Charter", "Iowan Old Style", "Georgia", "Times New Roman", serif;
        color: {INK}; letter-spacing: -0.012em; font-weight: 600;
      }}
      h3, h4 {{
        font-family: "Charter", "Iowan Old Style", "Georgia", serif;
        color: {INK}; letter-spacing: -0.005em; font-weight: 600;
      }}
      .kicker {{
        font-family: -apple-system, system-ui, sans-serif;
        color: {INK_SOFT}; font-size: 12px; letter-spacing: 0.16em;
        text-transform: uppercase; font-weight: 600;
      }}
      .dropcap::first-letter {{
        font-family: "Charter", "Iowan Old Style", "Georgia", serif;
        font-size: 3.6em; line-height: 0.85; font-weight: 700;
        float: left; padding: 6px 10px 0 0; color: {CHARCOAL};
      }}
      .lede {{
        font-size: 17.5px; line-height: 1.55; color: {INK};
        max-width: 740px;
      }}

      /* Risk hero card */
      .hero {{
        background: {PAPER}; border: 1px solid {RULE};
        border-radius: 4px; padding: 26px 30px; margin: 8px 0 18px 0;
      }}
      .hero .label {{
        font-family: -apple-system, system-ui, sans-serif;
        color: {INK_SOFT}; font-size: 11.5px; font-weight: 600;
        letter-spacing: 0.14em; text-transform: uppercase;
        margin-bottom: 2px;
      }}
      .hero .big {{
        font-family: "Charter", "Iowan Old Style", "Georgia", serif;
        font-size: 86px; font-weight: 700; line-height: 0.95;
        letter-spacing: -0.035em;
      }}
      .hero .tier {{
        font-family: -apple-system, system-ui, sans-serif;
        font-size: 14px; font-weight: 600; letter-spacing: 0.02em;
      }}
      .hero .interp {{
        font-family: "Charter", "Iowan Old Style", "Georgia", serif;
        color: {INK_SOFT}; font-size: 16.5px; line-height: 1.45;
        margin-top: 8px;
      }}

      /* Risk strip */
      .strip-band {{ font-family: -apple-system, system-ui, sans-serif;
                     font-size: 10.5px; color: {INK_SOFT}; }}

      /* Pull quote */
      .pull {{
        font-family: "Charter", "Iowan Old Style", "Georgia", serif;
        font-style: italic; color: {INK};
        font-size: 19px; line-height: 1.45;
        border-left: 3px solid {CHARCOAL};
        padding: 4px 0 4px 18px; margin: 6px 0 18px 0;
      }}
      .pull cite {{
        display: block; font-style: normal; font-size: 12px;
        color: {INK_SOFT}; letter-spacing: 0.04em;
        margin-top: 6px;
      }}

      /* Footer */
      .colophon {{
        font-family: -apple-system, system-ui, sans-serif;
        color: {INK_SOFT}; font-size: 12.5px; line-height: 1.6;
        border-top: 1px solid {RULE}; padding-top: 14px; margin-top: 28px;
      }}
      .colophon b {{ color: {INK}; }}
      .colophon a {{ color: {INK}; text-decoration: underline;
                     text-decoration-color: {RULE}; text-underline-offset: 3px; }}

      /* Sidebar polish */
      [data-testid="stSidebar"] h3 {{ font-size: 14px; }}
      .preset-row p {{ margin: 0; padding: 0; font-size: 12px; color: {INK_SOFT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_FONT = "Charter, Iowan Old Style, Georgia, serif"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "cvd_risk_v1.joblib")
META_PATH = os.path.join(HERE, "models", "training_metadata.json")
SCALER_PATH = os.path.join(HERE, "models", "scaler.json")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return (bundle["pipeline"], bundle["isotonic"],
            bundle["feature_cols"], bundle["feature_medians"])


@st.cache_resource
def load_metadata():
    with open(META_PATH) as fh:
        return json.load(fh)


@st.cache_resource
def load_scaler():
    with open(SCALER_PATH) as fh:
        return json.load(fh)


def predict_calibrated(X: pd.DataFrame) -> np.ndarray:
    raw = pipeline.predict_proba(X)[:, 1]
    return isotonic.predict(raw)


pipeline, isotonic, FEATURE_COLS, FEATURE_MEDIANS = load_model()
META = load_metadata()
SCALER = load_scaler()


# ---------------------------------------------------------------------------
# Feature catalogue — drives the sidebar UI and labels each control with its
# HealthKit identifier, mirroring src/healthkit_schema.py.
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
    group: str


INPUTS: list[Input] = [
    Input("age",                "Age",                          "HKCharacteristicTypeIdentifierDateOfBirth",
          "years",  30, 80, 1.0, "Demographics"),
    Input("biological_sex",     "Biological sex",               "HKCharacteristicTypeIdentifierBiologicalSex",
          "enum",   0, 1, 1.0, "Demographics"),
    Input("avg_daily_step_count",     "Average daily steps",     "HKQuantityTypeIdentifierStepCount",
          "steps",  500, 25000, 100.0, "Activity"),
    Input("avg_daily_active_energy",  "Active energy",           "HKQuantityTypeIdentifierActiveEnergyBurned",
          "kcal",   50, 1500, 10.0, "Activity"),
    Input("avg_daily_exercise_min",   "Exercise minutes",        "HKQuantityTypeIdentifierAppleExerciseTime",
          "min",    0, 180, 1.0, "Activity"),
    Input("peak_intensity",           "Peak intensity",          "HKQuantityTypeIdentifierActiveEnergyBurned",
          "kcal",   50, 2500, 10.0, "Activity"),
    Input("activity_regularity",      "Activity regularity",     "derived",
          "0–1",    0.1, 1.0, 0.01, "Activity"),
    Input("low_activity_day_ratio",   "Low-activity day ratio",  "derived",
          "0–1",    0.0, 1.0, 0.01, "Activity"),
    Input("sedentary_minutes",        "Sedentary minutes",       "HKQuantityTypeIdentifierAppleStandTime",
          "min",    100, 1300, 5.0, "Activity"),
    Input("avg_sleep_hours",          "Average sleep",           "HKCategoryTypeIdentifierSleepAnalysis",
          "hours",  3.0, 12.0, 0.1, "Sleep"),
    Input("sleep_regularity",         "Sleep regularity",        "derived",
          "0–1",    0.05, 1.0, 0.01, "Sleep"),
    Input("circadian_light_exposure", "Daytime light exposure",  "derived (ambient light)",
          "lux",    1000, 600000, 1000.0, "Circadian"),
]

INPUT_BY_KEY = {inp.key: inp for inp in INPUTS}
SEX_LABELS = ["Female", "Male"]


# ---------------------------------------------------------------------------
# Three named profiles that snap the sidebar to realistic values.
# These are illustrative profiles, not real people — they exist so a visitor
# can see the model behave on someone other than the cohort median.
# ---------------------------------------------------------------------------
PRESETS = {
    "Maya, 42 — runs three mornings a week": {
        "age": 42, "biological_sex": "Female",
        "avg_daily_step_count": 11800, "avg_daily_active_energy": 620,
        "avg_daily_exercise_min": 52, "peak_intensity": 1100,
        "activity_regularity": 0.88, "low_activity_day_ratio": 0.10,
        "sedentary_minutes": 540, "avg_sleep_hours": 7.6,
        "sleep_regularity": 0.55, "circadian_light_exposure": 180000,
    },
    "An average NHANES adult": {
        "age": 51, "biological_sex": "Female",
        "avg_daily_step_count": 6650, "avg_daily_active_energy": 360,
        "avg_daily_exercise_min": 36, "peak_intensity": 510,
        "activity_regularity": 0.79, "low_activity_day_ratio": 0.25,
        "sedentary_minutes": 750, "avg_sleep_hours": 6.8,
        "sleep_regularity": 0.31, "circadian_light_exposure": 119000,
    },
    "Daniel, 58 — desk job, weekend warrior": {
        "age": 58, "biological_sex": "Male",
        "avg_daily_step_count": 4200, "avg_daily_active_energy": 240,
        "avg_daily_exercise_min": 14, "peak_intensity": 380,
        "activity_regularity": 0.55, "low_activity_day_ratio": 0.45,
        "sedentary_minutes": 920, "avg_sleep_hours": 5.9,
        "sleep_regularity": 0.18, "circadian_light_exposure": 75000,
    },
}


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
def _native(key, value):
    """Coerce a numeric default to the widget's native type."""
    if key == "biological_sex":
        return value if isinstance(value, str) else SEX_LABELS[int(value)]
    if key in {"age", "avg_daily_step_count", "avg_daily_active_energy",
               "avg_daily_exercise_min", "peak_intensity",
               "sedentary_minutes", "circadian_light_exposure"}:
        return int(round(value))
    return float(value)


def _seed_defaults(values: dict):
    for inp in INPUTS:
        st.session_state[inp.key] = _native(inp.key, values[inp.key])


for inp in INPUTS:
    if inp.key not in st.session_state:
        st.session_state[inp.key] = _native(inp.key, FEATURE_MEDIANS[inp.key])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f"<div style='font-family:Charter,serif; font-size:22px; "
    f"font-weight:700; color:{INK}; margin-bottom:0.4rem;'>"
    f"A New Framingham</div>"
    f"<div class='kicker' style='margin-bottom:18px;'>1948 · NHANES · today</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("#### Pick a profile")
st.sidebar.caption(
    "Three quick portraits of real-feeling people. Tap one to load their numbers, "
    "then nudge the sliders to match your own life."
)
preset_choice = st.sidebar.selectbox(
    "Profile", list(PRESETS.keys()), label_visibility="collapsed",
    key="_preset_select",
)
b1, b2 = st.sidebar.columns(2)
with b1:
    if st.button("Load profile", width="stretch"):
        _seed_defaults(PRESETS[preset_choice])
        st.rerun()
with b2:
    if st.button("Cohort median", width="stretch"):
        for inp in INPUTS:
            st.session_state[inp.key] = _native(inp.key, FEATURE_MEDIANS[inp.key])
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("#### Your numbers")
st.sidebar.caption(
    "Each control corresponds to a HealthKit identifier the companion iOS "
    "app reads off an Apple Watch."
)

with st.sidebar.expander("Demographics", expanded=True):
    st.slider("Age (years)", 30, 80, key="age")
    st.radio("Biological sex", SEX_LABELS, horizontal=True, key="biological_sex")

with st.sidebar.expander("Activity", expanded=True):
    st.slider("Daily steps", 500, 25000, step=100, key="avg_daily_step_count",
              help="HKQuantityTypeIdentifierStepCount · 7-day mean")
    st.slider("Active energy (kcal/day)", 50, 1500, step=10,
              key="avg_daily_active_energy",
              help="HKQuantityTypeIdentifierActiveEnergyBurned")
    st.slider("Exercise minutes/day", 0, 180, key="avg_daily_exercise_min",
              help="HKQuantityTypeIdentifierAppleExerciseTime")
    st.slider("Peak day intensity (kcal)", 50, 2500, step=10,
              key="peak_intensity")
    st.slider("Activity regularity", 0.10, 1.00, step=0.01,
              key="activity_regularity",
              help="Higher = more consistent day-to-day movement")
    st.slider("Low-activity day fraction", 0.00, 1.00, step=0.01,
              key="low_activity_day_ratio")
    st.slider("Sedentary minutes/day", 100, 1300, step=5,
              key="sedentary_minutes",
              help="HKQuantityTypeIdentifierAppleStandTime (inverse)")

with st.sidebar.expander("Sleep", expanded=True):
    st.slider("Average sleep (hours)", 3.0, 12.0, step=0.1,
              key="avg_sleep_hours",
              help="HKCategoryTypeIdentifierSleepAnalysis")
    st.slider("Sleep regularity", 0.05, 1.00, step=0.01,
              key="sleep_regularity",
              help="Higher = more consistent nightly sleep")

with st.sidebar.expander("Circadian", expanded=False):
    st.slider("Daytime light exposure (lux)", 1000, 600000, step=1000,
              key="circadian_light_exposure")


# ---------------------------------------------------------------------------
# Build the feature vector and predict
# ---------------------------------------------------------------------------
features = {col: float(st.session_state[col])
            for col in FEATURE_COLS if col != "biological_sex"}
features["biological_sex"] = float(SEX_LABELS.index(st.session_state["biological_sex"]))
X = pd.DataFrame([[features[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
risk = float(predict_calibrated(X)[0])


def categorise(p: float):
    if p < 0.05:
        return "Low", SAGE, "Below 5%. Your current pattern is among the calmest in the cohort."
    if p < 0.10:
        return "Borderline", OCHRE, "Between 5% and 10%. Worth watching the trend, not yet worth worrying."
    if p < 0.20:
        return "Intermediate", OCHRE, "Between 10% and 20%. Lifestyle levers can move this materially."
    return "High", CRIMSON, "Above 20%. Worth a real conversation with a clinician."


category, color, interp = categorise(risk)


# ---------------------------------------------------------------------------
# Editorial header
# ---------------------------------------------------------------------------
st.markdown(
    f"<div class='kicker' style='color:{CRIMSON};'>"
    f"FRAMINGHAM · MASS · 1948 → YOUR WRIST · TODAY</div>",
    unsafe_allow_html=True,
)
st.markdown("<h1 style='margin-top:0.2rem;'>A New Framingham</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<div class='lede dropcap'>"
    "In 1948, 5,209 residents of Framingham, Massachusetts agreed to be "
    "measured every two years for the rest of their lives. The cholesterol, "
    "blood pressure, and smoking habits they wrote down on those clipboards "
    "became the foundation of modern cardiology — the basis of the ten-year "
    "risk score a doctor still pulls up before deciding whether to put you "
    "on a statin."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='lede' style='margin-top:14px;'>"
    "Most of those signals — and a few the 1948 study could not have imagined "
    "— now live on a watch. <b>This page asks how far you can get toward the "
    "Framingham number without a needle.</b> The model was trained on 2,967 "
    f"NHANES wearable participants, then calibrated so the percentage you see "
    f"reads as an actual probability. AUC <b>{META['metrics']['auc_cv5']:.2f}</b>, "
    f"Brier <b>{META['metrics']['brier_cv5_calibrated']:.2f}</b> after isotonic "
    "calibration."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Hero risk card + horizontal risk strip
# ---------------------------------------------------------------------------
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

hero_left, hero_right = st.columns([1.05, 1])

with hero_left:
    st.markdown(
        f"""
        <div class="hero">
          <div class="label">Estimated 10-year CVD risk</div>
          <div class="big" style="color:{color};">{risk*100:.1f}%</div>
          <div class="tier" style="color:{color}; margin-top:6px;">{category.upper()}</div>
          <div class="interp">{interp}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    # Horizontal risk strip — four clinical bands with the user's mark.
    strip = go.Figure()
    bands = [
        (0,  5,  "#E5EFE7", "Low",          SAGE),
        (5,  10, "#F2E9D6", "Borderline",   OCHRE),
        (10, 20, "#EDDDC1", "Intermediate", OCHRE),
        (20, 50, "#EBC9C5", "High",         CRIMSON),
    ]
    for lo, hi, band_color, label, _ in bands:
        strip.add_shape(
            type="rect",
            x0=lo, x1=hi, y0=0, y1=1,
            line=dict(width=0), fillcolor=band_color, layer="below",
        )
        strip.add_annotation(
            x=(lo + hi) / 2, y=1.18, xref="x", yref="y",
            text=label, showarrow=False,
            font=dict(family="-apple-system, system-ui",
                      size=10, color=INK_SOFT),
        )

    risk_pct = max(0.6, min(49.4, risk * 100))
    strip.add_shape(
        type="line",
        x0=risk_pct, x1=risk_pct, y0=-0.18, y1=1.05,
        line=dict(color=color, width=3),
    )
    strip.add_annotation(
        x=risk_pct, y=-0.42, xref="x", yref="y",
        text=f"<b>{risk*100:.1f}%</b>",
        showarrow=False,
        font=dict(family=PLOT_FONT, size=15, color=color),
    )
    strip.add_annotation(
        x=10, y=-0.72, xref="x", yref="y",
        text="10% — the line clinicians draw",
        showarrow=False,
        font=dict(family="-apple-system, system-ui",
                  size=10, color=INK_SOFT),
    )

    strip.update_layout(
        height=200, margin=dict(l=14, r=14, t=24, b=10),
        plot_bgcolor=PAPER, paper_bgcolor=PAPER,
        xaxis=dict(range=[0, 50],
                   tickvals=[0, 5, 10, 20, 30, 50],
                   ticktext=["0%", "5%", "10%", "20%", "30%", "50%"],
                   tickfont=dict(family="-apple-system, system-ui",
                                 size=10, color=INK_SOFT),
                   showgrid=False, zeroline=False, showline=False, ticks="outside"),
        yaxis=dict(range=[-0.9, 1.4], visible=False),
    )
    st.plotly_chart(strip, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Local sensitivity — "what's pulling the number?"
# ---------------------------------------------------------------------------
@st.cache_data
def cohort_spreads():
    """For each feature: (slider_min, slider_max, ±step for sensitivity).
    The step is one training-set standard deviation so deltas are comparable
    across features with wildly different units."""
    sd_by_feature = dict(zip(SCALER["feature_order"], SCALER["scale"]))
    ranges = {inp.key: (inp.min, inp.max) for inp in INPUTS}
    return {k: (ranges[k][0], ranges[k][1], sd_by_feature[k])
            for k in FEATURE_COLS}


def local_sensitivity(features: dict) -> pd.DataFrame:
    base_vec = pd.DataFrame([[features[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
    base = float(predict_calibrated(base_vec)[0])
    spreads = cohort_spreads()
    rows = []
    for col in FEATURE_COLS:
        lo, hi, step = spreads[col]
        v = features[col]
        up_v = min(hi, v + step)
        dn_v = max(lo, v - step)
        up = base_vec.copy(); up[col] = up_v
        dn = base_vec.copy(); dn[col] = dn_v
        p_up = float(predict_calibrated(up)[0])
        p_dn = float(predict_calibrated(dn)[0])
        rows.append({
            "feature": col,
            "label": INPUT_BY_KEY[col].label,
            "delta_up":   p_up - base,
            "delta_down": p_dn - base,
            "swing":      abs(p_up - base) + abs(p_dn - base),
        })
    return pd.DataFrame(rows).sort_values("swing", ascending=False)


sens = local_sensitivity(features)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("---")

left, right = st.columns([1.1, 1])

with left:
    st.markdown("<div class='kicker'>SENSITIVITY</div>", unsafe_allow_html=True)
    st.markdown("### What's pulling the number?")
    st.markdown(
        f"<div style='color:{INK_SOFT}; font-size:14.5px; line-height:1.5; "
        f"max-width:560px;'>"
        "Each bar shows how much your estimated risk would shift if a single "
        "signal moved one cohort-standard-deviation in either direction, with "
        "everything else held fixed. Long bars are the levers worth pulling."
        "</div>",
        unsafe_allow_html=True,
    )

    top = sens.head(6).iloc[::-1]
    bar = go.Figure()
    bar.add_trace(go.Bar(
        y=top["label"], x=top["delta_down"] * 100,
        orientation="h", name="If lower",
        marker=dict(color=SAGE, line=dict(width=0)),
        hovertemplate="%{x:+.1f} pp<extra>If lower</extra>",
    ))
    bar.add_trace(go.Bar(
        y=top["label"], x=top["delta_up"] * 100,
        orientation="h", name="If higher",
        marker=dict(color=CRIMSON, line=dict(width=0)),
        hovertemplate="%{x:+.1f} pp<extra>If higher</extra>",
    ))
    bar.update_layout(
        barmode="relative", height=320,
        margin=dict(l=10, r=10, t=12, b=10),
        xaxis=dict(title="Change in predicted risk (percentage points)",
                   gridcolor=RULE, zerolinecolor=INK_SOFT,
                   tickfont=dict(family="-apple-system, system-ui", size=11)),
        yaxis=dict(tickfont=dict(family="-apple-system, system-ui", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                    font=dict(family="-apple-system, system-ui", size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family=PLOT_FONT, color=INK),
    )
    bar.add_vline(x=0, line_color=INK_SOFT, line_width=1)
    st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})

with right:
    st.markdown("<div class='kicker'>POSITION</div>", unsafe_allow_html=True)
    st.markdown("### Where you sit in the cohort")
    st.markdown(
        f"<div style='color:{INK_SOFT}; font-size:14.5px; line-height:1.5; "
        f"max-width:520px;'>"
        "Your inputs against the 2,967-person training cohort, expressed as "
        "percent change from the median. Greener rows mean you're better "
        "than typical on that signal; redder rows mean worse."
        "</div>",
        unsafe_allow_html=True,
    )
    base_rows = []
    for col in FEATURE_COLS:
        med = FEATURE_MEDIANS[col]
        cur = features[col]
        delta = 0.0 if med == 0 else (cur - med) / abs(med) * 100
        base_rows.append({
            "Feature": INPUT_BY_KEY[col].label,
            "Yours":   cur,
            "Cohort median": med,
            "Δ vs median": delta,
        })
    pct_df = pd.DataFrame(base_rows)
    st.dataframe(
        pct_df.style.format({
            "Yours": "{:,.1f}",
            "Cohort median": "{:,.1f}",
            "Δ vs median": "{:+.0f}%",
        }).background_gradient(subset=["Δ vs median"],
                               cmap="RdYlGn_r", vmin=-100, vmax=100),
        hide_index=True, width="stretch", height=320,
    )


# ---------------------------------------------------------------------------
# What-if sweep
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("<div class='kicker'>WHAT-IF</div>", unsafe_allow_html=True)
st.markdown("### Move one thing at a time")
st.markdown(
    f"<div style='color:{INK_SOFT}; font-size:14.5px; line-height:1.5; "
    f"max-width:720px;'>"
    "Sweep a single signal across its plausible range while the other "
    "eleven inputs stay exactly where you left them. The dashed line is "
    "the 10% threshold a clinician would treat as a real risk marker."
    "</div>",
    unsafe_allow_html=True,
)

knob_col, plot_col = st.columns([1, 2.4])
with knob_col:
    knob = st.selectbox(
        "Signal to sweep",
        options=[c for c in FEATURE_COLS if c not in ("biological_sex",)],
        format_func=lambda c: INPUT_BY_KEY[c].label,
        key="_sweep_knob",
    )

lo, hi, _ = cohort_spreads()[knob]
sweep_values = np.linspace(lo, hi, 60)
sweep_rows = []
for v in sweep_values:
    row = features.copy()
    row[knob] = float(v)
    Xv = pd.DataFrame([[row[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
    sweep_rows.append({"x": float(v),
                       "risk": float(predict_calibrated(Xv)[0]) * 100})
sweep_df = pd.DataFrame(sweep_rows)

with plot_col:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep_df["x"], y=sweep_df["risk"], mode="lines",
        line=dict(color=CHARCOAL, width=3),
        hovertemplate=f"{INPUT_BY_KEY[knob].label}: %{{x:,.1f}} "
                      f"{INPUT_BY_KEY[knob].unit}<br>Risk: %{{y:.1f}}%<extra></extra>",
    ))
    fig.add_hline(y=10, line_dash="dot", line_color=INK_SOFT,
                  annotation_text="10% clinical threshold",
                  annotation_position="top left",
                  annotation_font=dict(family="-apple-system, system-ui",
                                       size=10, color=INK_SOFT))
    fig.add_vline(x=features[knob], line_color=CRIMSON, line_width=2,
                  annotation_text="You",
                  annotation_position="top right",
                  annotation_font=dict(family="-apple-system, system-ui",
                                       size=11, color=CRIMSON))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=12, b=10),
        xaxis=dict(title=f"{INPUT_BY_KEY[knob].label} ({INPUT_BY_KEY[knob].unit})",
                   gridcolor=RULE,
                   tickfont=dict(family="-apple-system, system-ui", size=11)),
        yaxis=dict(title="Predicted 10-yr CVD risk (%)",
                   gridcolor=RULE,
                   tickfont=dict(family="-apple-system, system-ui", size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family=PLOT_FONT, color=INK),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Method + provenance
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div class='pull'>"
    "“The objective of this paper is to develop a sex-specific multivariable risk "
    "function… that can be used by primary-care doctors without the need for special "
    "equipment.”"
    "<cite>D'Agostino R.B. et al., 2008. <i>Circulation</i> 117(6).</cite>"
    "</div>",
    unsafe_allow_html=True,
)

mcol1, mcol2, mcol3 = st.columns(3)

with mcol1:
    st.markdown("<div class='kicker'>MODEL</div>", unsafe_allow_html=True)
    st.markdown(
        f"**{META['framework']}**  \n"
        f"{META['calibration']}  \n"
        f"{META['training_cohort']}  \n"
        f"N = {META['training_n']:,}"
    )

with mcol2:
    st.markdown("<div class='kicker'>CROSS-VALIDATION</div>", unsafe_allow_html=True)
    st.markdown(
        f"AUC (5-fold OOF): **{META['metrics']['auc_cv5']:.3f}**  \n"
        f"Brier — uncalibrated: {META['metrics']['brier_cv5_uncalibrated']:.3f}  \n"
        f"Brier — calibrated: **{META['metrics']['brier_cv5_calibrated']:.3f}**  \n"
        f"High-risk prevalence: {META['metrics']['high_risk_prevalence']*100:.1f}%"
    )

with mcol3:
    st.markdown("<div class='kicker'>ALSO ON YOUR PHONE</div>", unsafe_allow_html=True)
    st.markdown(
        "The same fitted tree ensemble is exported as `CVDRiskModel.mlmodel` "
        "(62 KB). A companion SwiftUI app reads the HealthKit identifiers "
        "shown next to each slider and runs the calibrated prediction "
        "entirely on-device."
    )


# ---------------------------------------------------------------------------
# Colophon
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="colophon">
      <p><b>Honest limitations.</b> NHANES 2011-2012 is a US cross-sectional
      sample; the Framingham coefficients themselves come from a
      non-representative cohort. The number above is a calibrated probability
      of being labelled <i>high-risk by Framingham</i> — one degree removed
      from actual cardiovascular events. The smoking covariate uses NHANES
      SMQ020 (“ever smoked ≥100 cigarettes”) rather than current-smoker
      status, which nudges former-smoker scores up. Ambient light is in the
      trained feature vector but isn't exposed by HealthKit; the iOS build
      substitutes the cohort median, mirroring the Python imputer. Treat the
      output as self-awareness, not a clinical decision.</p>

      <p><b>Data.</b> NHANES 2011-2012, U.S. CDC. Activity, sleep, and light
      derived from the PAXDAY accelerometer files. Labels: D'Agostino R.B.
      et al., 2008, “General Cardiovascular Risk Profile for Use in Primary
      Care,” <i>Circulation</i> 117(6).</p>

      <p style="margin-top:14px;">
        <b>Built by Arya Bhanushali.</b>
        Source &amp; methodology:
        <a href="https://github.com/aryabhanushali/newframingham" target="_blank">
          github.com/aryabhanushali/newframingham
        </a>.
        &nbsp;·&nbsp; Nothing on this page leaves your browser session.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
