"""
Heart — interactive 10-year cardiovascular risk estimate built in the visual
language of the Apple Health app.

Loads the same calibrated scikit-learn bundle the iOS app uses
(`models/cvd_risk_v1.joblib`) and predicts risk from activity, sleep, and
demographics — no cholesterol panel, no clinic visit. Nothing leaves the
browser session.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart · CVD Risk",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/aryabhanushali/newframingham",
        "Report a bug": "https://github.com/aryabhanushali/newframingham/issues",
        "About": (
            "Heart — 10-year cardiovascular risk from passive Apple Watch–style "
            "signals. A web companion to a Core ML / SwiftUI iOS prototype.\n\n"
            "Built by Arya Bhanushali · github.com/aryabhanushali/newframingham"
        ),
    },
)


# ---------------------------------------------------------------------------
# iOS-derived system palette + Apple-Health card CSS
# ---------------------------------------------------------------------------
INK         = "#FFFFFF"
SUB         = "#8E8E93"
HAIR        = "#2C2C2E"
SURFACE     = "#000000"
CARD        = "#1C1C1E"
CARD_HI     = "#2C2C2E"

RED         = "#FF3B30"
ORANGE      = "#FF9500"
YELLOW      = "#FFCC00"
GREEN       = "#34C759"
MINT        = "#00C7BE"
TEAL        = "#5AC8FA"
BLUE        = "#0A84FF"
INDIGO      = "#5E5CE6"
PURPLE      = "#BF5AF2"
PINK        = "#FF2D55"


SYSTEM_FONT = (
    '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", '
    '"Helvetica Neue", Helvetica, Arial, sans-serif'
)

st.markdown(
    f"""
    <style>
      /* Reset Streamlit page chrome */
      .main .block-container {{
        padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1180px;
      }}
      html, body, .stApp {{
        background: {SURFACE}; color: {INK};
        font-family: {SYSTEM_FONT};
        -webkit-font-smoothing: antialiased;
      }}
      [data-testid="stSidebar"] {{
        background: {CARD}; border-right: 1px solid {HAIR};
      }}
      [data-testid="stSidebar"] * {{ color: {INK}; }}
      [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {{
        color: {SUB} !important;
      }}

      h1, h2, h3, h4, h5 {{
        font-family: {SYSTEM_FONT};
        color: {INK}; letter-spacing: -0.022em; font-weight: 700;
      }}

      /* Standard Apple-Health section header */
      .sf-section {{
        display: flex; align-items: baseline; justify-content: space-between;
        margin: 28px 0 10px 0;
      }}
      .sf-section .title {{
        font-size: 22px; font-weight: 700; color: {INK}; letter-spacing: -0.02em;
      }}
      .sf-section .sub {{
        font-size: 13px; color: {SUB}; font-weight: 500;
      }}

      /* Generic card */
      .card {{
        background: {CARD}; border-radius: 18px; padding: 18px 20px;
        border: 1px solid {HAIR};
      }}
      .card.tall {{ min-height: 132px; }}

      /* Metric card (matches Apple Health "Highlights" tile) */
      .metric .label {{
        font-size: 11.5px; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; line-height: 1;
      }}
      .metric .value {{
        font-size: 30px; font-weight: 700; color: {INK};
        letter-spacing: -0.025em; margin-top: 10px; line-height: 1.05;
      }}
      .metric .unit {{
        color: {SUB}; font-weight: 500; font-size: 14px; margin-left: 4px;
      }}
      .metric .footnote {{
        color: {SUB}; font-size: 12px; margin-top: 8px;
      }}

      /* Hero (risk ring + caption) */
      .hero-card {{
        background: {CARD}; border: 1px solid {HAIR};
        border-radius: 22px; padding: 24px 26px;
      }}
      .hero-card .eyebrow {{
        font-size: 12px; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: {SUB};
      }}
      .hero-card .tier {{
        margin-top: 6px; font-size: 26px; font-weight: 700; letter-spacing: -0.02em;
      }}
      .hero-card .caption {{
        margin-top: 10px; color: {SUB}; font-size: 15px; line-height: 1.45;
        max-width: 440px;
      }}

      /* Pill chip */
      .chip {{
        display: inline-flex; align-items: center; gap: 6px;
        background: {CARD_HI}; border-radius: 999px;
        padding: 4px 11px; font-size: 12px; font-weight: 600;
        color: {INK};
      }}
      .chip .dot {{
        width: 6px; height: 6px; border-radius: 50%;
      }}

      /* Footer */
      .footnote-block {{
        color: {SUB}; font-size: 12.5px; line-height: 1.55;
        border-top: 1px solid {HAIR}; padding-top: 14px; margin-top: 28px;
      }}
      .footnote-block b {{ color: {INK}; }}
      .footnote-block a {{ color: {INK}; text-decoration: none;
                           border-bottom: 1px solid {HAIR}; }}

      /* Tooltip "?" icon should be visible on dark */
      [data-testid="stTooltipIcon"] svg {{ fill: {SUB}; }}

      /* Streamlit selectbox / dropdown on dark */
      .stSelectbox [data-baseweb] {{ color: {INK}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


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
# Feature catalogue (mirrors src/healthkit_schema.py)
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
    accent: str
    icon: str   # SF Symbols-ish unicode that's universally available


INPUTS: list[Input] = [
    Input("age",                     "Age",                "Birth date",
          "years", 30, 80, 1.0, BLUE,   "👤"),
    Input("biological_sex",          "Biological sex",     "Biological sex",
          "enum",  0, 1, 1.0, PURPLE, "⚥"),
    Input("avg_daily_step_count",    "Steps",              "StepCount",
          "/ day", 500, 25000, 100.0, GREEN, "👣"),
    Input("avg_daily_active_energy", "Active energy",      "ActiveEnergyBurned",
          "kcal",  50, 1500, 10.0, RED, "🔥"),
    Input("avg_daily_exercise_min",  "Exercise",           "AppleExerciseTime",
          "min",   0, 180, 1.0, GREEN, "🏃"),
    Input("peak_intensity",          "Peak intensity",     "ActiveEnergyBurned",
          "kcal",  50, 2500, 10.0, ORANGE, "⚡"),
    Input("activity_regularity",     "Activity regularity","derived",
          "0–1",   0.10, 1.00, 0.01, MINT, "📈"),
    Input("low_activity_day_ratio",  "Low-activity days",  "derived",
          "0–1",   0.00, 1.00, 0.01, YELLOW, "💤"),
    Input("sedentary_minutes",       "Sedentary",          "AppleStandTime",
          "min",   100, 1300, 5.0, ORANGE, "🪑"),
    Input("avg_sleep_hours",         "Sleep",              "SleepAnalysis",
          "h",     3.0, 12.0, 0.1, INDIGO, "🌙"),
    Input("sleep_regularity",        "Sleep regularity",   "derived",
          "0–1",   0.05, 1.00, 0.01, INDIGO, "🛏️"),
    Input("circadian_light_exposure","Daylight",           "ambient light",
          "lux",   1000, 600000, 1000.0, YELLOW, "☀️"),
]

INPUT_BY_KEY = {inp.key: inp for inp in INPUTS}
SEX_LABELS = ["Female", "Male"]


# ---------------------------------------------------------------------------
# Preset profiles for the "Try a profile" picker
# ---------------------------------------------------------------------------
PRESETS = {
    "Active adult — 42, runs 3×/week": {
        "age": 42, "biological_sex": "Female",
        "avg_daily_step_count": 11800, "avg_daily_active_energy": 620,
        "avg_daily_exercise_min": 52, "peak_intensity": 1100,
        "activity_regularity": 0.88, "low_activity_day_ratio": 0.10,
        "sedentary_minutes": 540, "avg_sleep_hours": 7.6,
        "sleep_regularity": 0.55, "circadian_light_exposure": 180000,
    },
    "Average NHANES adult": {
        "age": 51, "biological_sex": "Female",
        "avg_daily_step_count": 6650, "avg_daily_active_energy": 360,
        "avg_daily_exercise_min": 36, "peak_intensity": 510,
        "activity_regularity": 0.79, "low_activity_day_ratio": 0.25,
        "sedentary_minutes": 750, "avg_sleep_hours": 6.8,
        "sleep_regularity": 0.31, "circadian_light_exposure": 119000,
    },
    "Sedentary adult — 58, desk job": {
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
# Sidebar — feels like an Apple Settings sheet
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f"<div style='font-size:22px; font-weight:700; letter-spacing:-0.02em; "
    f"margin: 4px 0 2px 0;'>Health</div>"
    f"<div style='font-size:13px; color:{SUB}; margin-bottom:14px;'>"
    f"Enter your numbers</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"<div style='font-size:11.5px; font-weight:700; letter-spacing:0.08em; "
    f"text-transform:uppercase; color:{SUB}; margin-bottom:6px;'>"
    f"Try a profile</div>",
    unsafe_allow_html=True,
)
preset_choice = st.sidebar.selectbox(
    "Profile", list(PRESETS.keys()), label_visibility="collapsed",
    key="_preset_select",
)
b1, b2 = st.sidebar.columns(2)
with b1:
    if st.button("Load", width="stretch"):
        _seed_defaults(PRESETS[preset_choice])
        st.rerun()
with b2:
    if st.button("Reset", width="stretch"):
        for inp in INPUTS:
            st.session_state[inp.key] = _native(inp.key, FEATURE_MEDIANS[inp.key])
        st.rerun()

st.sidebar.markdown(
    f"<div style='height:1px; background:{HAIR}; margin:14px 0 10px 0;'></div>",
    unsafe_allow_html=True,
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
        return "Low", GREEN, "Continue your current pattern. Your activity and sleep signals are favourable."
    if p < 0.10:
        return "Borderline", YELLOW, "On the line. The same habits a year from now will tell the real story."
    if p < 0.20:
        return "Intermediate", ORANGE, "Above the clinical threshold. Lifestyle levers can move this meaningfully."
    return "High", RED, "Worth a real conversation with a clinician."


tier_name, tier_color, tier_caption = categorise(risk)


# ---------------------------------------------------------------------------
# Top bar — large Health-app-style header
# ---------------------------------------------------------------------------
today_str = date.today().strftime("%A, %B %-d")
top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown(
        f"<div style='font-size:40px; font-weight:700; letter-spacing:-0.025em; "
        f"line-height:1.05;'>Heart</div>"
        f"<div style='color:{SUB}; font-size:15px; margin-top:2px;'>"
        f"10-year cardiovascular outlook · {today_str}</div>",
        unsafe_allow_html=True,
    )
with top_right:
    st.markdown(
        f"<div style='text-align:right; padding-top:14px;'>"
        f"<span class='chip'><span class='dot' style='background:{tier_color};'></span>"
        f"{tier_name} risk</span></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero — Apple Activity-ring style risk ring + caption card
# ---------------------------------------------------------------------------
def ring_svg(percent: float, color: str, size: int = 240, stroke: int = 22) -> str:
    """Render an Apple-Activity-style ring as inline SVG."""
    r = (size / 2) - (stroke / 2) - 4
    cx = cy = size / 2
    circumference = 2 * np.pi * r
    pct = max(0.0, min(percent, 0.999))
    dash = circumference * pct
    return f"""
    <svg viewBox="0 0 {size} {size}" width="{size}" height="{size}">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
              stroke="{HAIR}" stroke-width="{stroke}" />
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
              stroke="{color}" stroke-width="{stroke}"
              stroke-linecap="round"
              stroke-dasharray="{dash:.2f} {circumference:.2f}"
              transform="rotate(-90 {cx} {cy})" />
      <text x="{cx}" y="{cy - 6}" text-anchor="middle"
            font-family='{SYSTEM_FONT}'
            font-size="{int(size*0.20)}" font-weight="700" fill="{INK}"
            letter-spacing="-2">{percent*100:.1f}%</text>
      <text x="{cx}" y="{cy + size*0.13}" text-anchor="middle"
            font-family='{SYSTEM_FONT}'
            font-size="{int(size*0.055)}" font-weight="600"
            fill="{SUB}" letter-spacing="2">10-YR CVD RISK</text>
    </svg>
    """


st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
hero_l, hero_r = st.columns([1, 1.3])
with hero_l:
    st.markdown(
        f"<div class='hero-card' style='display:flex; justify-content:center; "
        f"align-items:center; min-height:300px;'>"
        f"{ring_svg(risk, tier_color)}"
        f"</div>",
        unsafe_allow_html=True,
    )

with hero_r:
    st.markdown(
        f"""
        <div class='hero-card' style='min-height:300px;'>
          <div class='eyebrow'>Today's reading</div>
          <div class='tier' style='color:{tier_color};'>{tier_name}</div>
          <div class='caption'>{tier_caption}</div>
          <div style='margin-top:22px; display:grid; grid-template-columns:1fr 1fr; gap:14px;'>
            <div>
              <div style='color:{SUB}; font-size:11.5px; font-weight:700;
                          letter-spacing:0.06em; text-transform:uppercase;'>Cohort AUC</div>
              <div style='font-size:24px; font-weight:700; margin-top:4px;'>
                {META['metrics']['auc_cv5']:.3f}</div>
            </div>
            <div>
              <div style='color:{SUB}; font-size:11.5px; font-weight:700;
                          letter-spacing:0.06em; text-transform:uppercase;'>Calibrated Brier</div>
              <div style='font-size:24px; font-weight:700; margin-top:4px;'>
                {META['metrics']['brier_cv5_calibrated']:.3f}</div>
            </div>
            <div>
              <div style='color:{SUB}; font-size:11.5px; font-weight:700;
                          letter-spacing:0.06em; text-transform:uppercase;'>Trained on</div>
              <div style='font-size:24px; font-weight:700; margin-top:4px;'>
                {META['training_n']:,}</div>
              <div style='color:{SUB}; font-size:12px;'>NHANES adults · ages 30–74</div>
            </div>
            <div>
              <div style='color:{SUB}; font-size:11.5px; font-weight:700;
                          letter-spacing:0.06em; text-transform:uppercase;'>Threshold</div>
              <div style='font-size:24px; font-weight:700; margin-top:4px;'>
                10%</div>
              <div style='color:{SUB}; font-size:12px;'>clinical action line</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Highlights — Apple Health-style metric tiles for the inputs that matter most
# ---------------------------------------------------------------------------
st.markdown(
    f"<div class='sf-section'>"
    f"<div class='title'>Highlights</div>"
    f"<div class='sub'>From your Apple Watch–style inputs</div>"
    f"</div>",
    unsafe_allow_html=True,
)


def metric_card(inp: Input, value_str: str, sub: str = ""):
    return (
        f"<div class='card tall metric'>"
        f"  <div style='display:flex; gap:8px; align-items:center;'>"
        f"    <span style='font-size:14px;'>{inp.icon}</span>"
        f"    <span class='label' style='color:{inp.accent};'>{inp.label}</span>"
        f"  </div>"
        f"  <div class='value'>{value_str}<span class='unit'>{inp.unit}</span></div>"
        f"  <div class='footnote'>{sub}</div>"
        f"</div>"
    )


def fmt(v, kind):
    if kind == "int":      return f"{int(round(v)):,}"
    if kind == "float1":   return f"{v:.1f}"
    if kind == "float2":   return f"{v:.2f}"
    if kind == "kcal":     return f"{int(round(v)):,}"
    return str(v)


sex_word = "Male" if features["biological_sex"] == 1 else "Female"

# Two rows of three.
r1c1, r1c2, r1c3 = st.columns(3, gap="medium")
with r1c1:
    st.markdown(metric_card(
        INPUT_BY_KEY["avg_daily_step_count"],
        fmt(features["avg_daily_step_count"], "int"),
        "7-day average",
    ), unsafe_allow_html=True)
with r1c2:
    st.markdown(metric_card(
        INPUT_BY_KEY["avg_daily_active_energy"],
        fmt(features["avg_daily_active_energy"], "kcal"),
        "7-day average",
    ), unsafe_allow_html=True)
with r1c3:
    st.markdown(metric_card(
        INPUT_BY_KEY["avg_daily_exercise_min"],
        fmt(features["avg_daily_exercise_min"], "int"),
        "Minutes above brisk-walk intensity",
    ), unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
r2c1, r2c2, r2c3 = st.columns(3, gap="medium")
with r2c1:
    st.markdown(metric_card(
        INPUT_BY_KEY["avg_sleep_hours"],
        fmt(features["avg_sleep_hours"], "float1"),
        "Mean nightly duration",
    ), unsafe_allow_html=True)
with r2c2:
    st.markdown(metric_card(
        INPUT_BY_KEY["sedentary_minutes"],
        fmt(features["sedentary_minutes"], "int"),
        "Daily sedentary minutes",
    ), unsafe_allow_html=True)
with r2c3:
    st.markdown(
        f"<div class='card tall metric'>"
        f"  <div style='display:flex; gap:8px; align-items:center;'>"
        f"    <span style='font-size:14px;'>👤</span>"
        f"    <span class='label' style='color:{BLUE};'>Profile</span>"
        f"  </div>"
        f"  <div class='value'>{int(features['age'])}<span class='unit'>· {sex_word}</span></div>"
        f"  <div class='footnote'>Age, biological sex</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Local sensitivity
# ---------------------------------------------------------------------------
@st.cache_data
def cohort_spreads():
    """For each feature: (slider_min, slider_max, ±step for sensitivity)."""
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
            "label":   INPUT_BY_KEY[col].label,
            "delta_up":   p_up - base,
            "delta_down": p_dn - base,
            "swing":      abs(p_up - base) + abs(p_dn - base),
        })
    return pd.DataFrame(rows).sort_values("swing", ascending=False)


sens = local_sensitivity(features)

st.markdown(
    f"<div class='sf-section'>"
    f"<div class='title'>What moves this</div>"
    f"<div class='sub'>Per-feature sensitivity, ±1 cohort SD</div>"
    f"</div>",
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 1], gap="medium")

with left:
    top = sens.head(6).iloc[::-1]
    bar = go.Figure()
    bar.add_trace(go.Bar(
        y=top["label"], x=top["delta_down"] * 100,
        orientation="h", name="If lower",
        marker=dict(color=GREEN, line=dict(width=0)),
        hovertemplate="%{x:+.1f} pp<extra>If lower</extra>",
    ))
    bar.add_trace(go.Bar(
        y=top["label"], x=top["delta_up"] * 100,
        orientation="h", name="If higher",
        marker=dict(color=RED, line=dict(width=0)),
        hovertemplate="%{x:+.1f} pp<extra>If higher</extra>",
    ))
    bar.update_layout(
        barmode="relative", height=340,
        margin=dict(l=10, r=10, t=14, b=10),
        xaxis=dict(title="Change in predicted risk (percentage points)",
                   gridcolor=HAIR, zerolinecolor=SUB, color=SUB,
                   tickfont=dict(family=SYSTEM_FONT, size=11)),
        yaxis=dict(color=INK,
                   tickfont=dict(family=SYSTEM_FONT, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                    font=dict(family=SYSTEM_FONT, size=11, color=INK)),
        plot_bgcolor=CARD, paper_bgcolor=CARD,
        font=dict(family=SYSTEM_FONT, color=INK),
    )
    bar.add_vline(x=0, line_color=SUB, line_width=1)
    st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})

with right:
    base_rows = []
    for col in FEATURE_COLS:
        med = FEATURE_MEDIANS[col]
        cur = features[col]
        delta = 0.0 if med == 0 else (cur - med) / abs(med) * 100
        base_rows.append({
            "Feature": INPUT_BY_KEY[col].label,
            "You":     cur,
            "Median":  med,
            "Δ":       delta,
        })
    pct_df = pd.DataFrame(base_rows)
    st.dataframe(
        pct_df.style.format({
            "You": "{:,.1f}",
            "Median": "{:,.1f}",
            "Δ": "{:+.0f}%",
        }).background_gradient(subset=["Δ"], cmap="RdYlGn_r", vmin=-100, vmax=100),
        hide_index=True, width="stretch", height=340,
    )


# ---------------------------------------------------------------------------
# What-if sweep
# ---------------------------------------------------------------------------
st.markdown(
    f"<div class='sf-section'>"
    f"<div class='title'>If one thing changed</div>"
    f"<div class='sub'>Sweep a single signal; everything else stays where you set it</div>"
    f"</div>",
    unsafe_allow_html=True,
)

knob_col, plot_col = st.columns([1, 2.4], gap="medium")
with knob_col:
    knob = st.selectbox(
        "Signal",
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
        line=dict(color=INPUT_BY_KEY[knob].accent, width=3),
        hovertemplate=f"{INPUT_BY_KEY[knob].label}: %{{x:,.1f}} "
                      f"{INPUT_BY_KEY[knob].unit}<br>Risk: %{{y:.1f}}%<extra></extra>",
    ))
    fig.add_hline(y=10, line_dash="dot", line_color=SUB,
                  annotation_text="10% clinical threshold",
                  annotation_position="top left",
                  annotation_font=dict(family=SYSTEM_FONT, size=10, color=SUB))
    fig.add_vline(x=features[knob], line_color=RED, line_width=2,
                  annotation_text="You",
                  annotation_position="top right",
                  annotation_font=dict(family=SYSTEM_FONT, size=11, color=RED))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=12, b=10),
        xaxis=dict(title=f"{INPUT_BY_KEY[knob].label} ({INPUT_BY_KEY[knob].unit})",
                   gridcolor=HAIR, color=SUB,
                   tickfont=dict(family=SYSTEM_FONT, size=11)),
        yaxis=dict(title="Predicted 10-yr CVD risk (%)",
                   gridcolor=HAIR, color=SUB,
                   tickfont=dict(family=SYSTEM_FONT, size=11)),
        plot_bgcolor=CARD, paper_bgcolor=CARD,
        font=dict(family=SYSTEM_FONT, color=INK),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# About this estimate
# ---------------------------------------------------------------------------
st.markdown(
    f"<div class='sf-section'>"
    f"<div class='title'>About this score</div>"
    f"<div class='sub'>Model, calibration, and what runs on iPhone</div>"
    f"</div>",
    unsafe_allow_html=True,
)

a, b, c = st.columns(3, gap="medium")
with a:
    st.markdown(
        f"<div class='card'>"
        f"<div class='label' style='color:{BLUE}; font-size:11.5px; "
        f"font-weight:700; letter-spacing:0.06em; text-transform:uppercase;'>Model</div>"
        f"<div style='margin-top:10px; line-height:1.55;'>"
        f"<b>{META['framework']}</b><br>"
        f"{META['calibration']}<br>"
        f"{META['training_cohort']}<br>"
        f"N = {META['training_n']:,}"
        f"</div></div>",
        unsafe_allow_html=True,
    )
with b:
    st.markdown(
        f"<div class='card'>"
        f"<div class='label' style='color:{ORANGE}; font-size:11.5px; "
        f"font-weight:700; letter-spacing:0.06em; text-transform:uppercase;'>Validation</div>"
        f"<div style='margin-top:10px; line-height:1.7;'>"
        f"AUC (5-fold OOF): <b>{META['metrics']['auc_cv5']:.3f}</b><br>"
        f"Brier — uncalibrated: {META['metrics']['brier_cv5_uncalibrated']:.3f}<br>"
        f"Brier — calibrated: <b>{META['metrics']['brier_cv5_calibrated']:.3f}</b><br>"
        f"High-risk prevalence: {META['metrics']['high_risk_prevalence']*100:.1f}%"
        f"</div></div>",
        unsafe_allow_html=True,
    )
with c:
    st.markdown(
        f"<div class='card'>"
        f"<div class='label' style='color:{PINK}; font-size:11.5px; "
        f"font-weight:700; letter-spacing:0.06em; text-transform:uppercase;'>On iPhone</div>"
        f"<div style='margin-top:10px; line-height:1.55;'>"
        f"The same fitted tree ensemble is exported as a 62 KB "
        f"<code>CVDRiskModel.mlmodel</code>. A companion SwiftUI app reads the "
        f"HealthKit identifiers shown next to each slider and runs the "
        f"calibrated prediction entirely on-device."
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Colophon
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="footnote-block">
      <p><b>Not a medical device.</b> NHANES 2011-2012 is a US cross-sectional
      sample; the Framingham coefficients themselves come from a
      non-representative cohort. The number above is a calibrated probability
      of being labelled <i>high-risk by Framingham</i> — one degree removed
      from actual cardiovascular events. The smoking covariate uses NHANES
      SMQ020 (“ever smoked ≥100 cigarettes”) rather than current-smoker
      status, which nudges former-smoker scores up. Ambient light is in the
      trained feature vector but isn't exposed by HealthKit; the iOS build
      substitutes the cohort median. Treat the output as self-awareness, not
      a clinical decision.</p>

      <p><b>Data.</b> NHANES 2011-2012, U.S. CDC.
      <b>Labels.</b> D'Agostino R.B. et al., 2008, “General Cardiovascular Risk
      Profile for Use in Primary Care,” <i>Circulation</i> 117(6).</p>

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
