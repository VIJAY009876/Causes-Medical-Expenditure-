"""
pages/Statistics.py — Statistical Model Results

FIX: GLM results cached to disk as CSV after first run.
     Subsequent page loads read from cache — NO re-fitting.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Statistics | NSSO", layout="wide")
st.title("📐 Statistical Models — Weighted GLM Results")

st.markdown("""
All models use **survey multiplier weights (`mult1`)** for population-representative estimates.

| Outcome | Model |
|---------|-------|
| OOPE (Out-of-Pocket Expenditure) | Weighted Gamma GLM (log link) |
| CHE / Poverty Line / Distress    | Weighted Binomial GLM (logit) |
""")

BASE     = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE, "data", "processed_data")
STAT_DIR = os.path.join(BASE, "models", "stat_results")   # GLM cache folder

MODEL_KEYS = [
    "oope_hosp", "oope_non_hosp",
    "che_hosp",  "che_non_hosp",
    "pl_hosp",   "pl_non_hosp",
    "distress_hosp", "distress_non_hosp",
]

# ── Load data (cached in memory) ──────────────────────────────────
@st.cache_data
def load_data():
    hosp     = pd.read_csv(os.path.join(DATA_DIR, "hospital_model.csv"),     index_col=0)
    non_hosp = pd.read_csv(os.path.join(DATA_DIR, "non_hospital_model.csv"), index_col=0)
    hosp.fillna(0, inplace=True)
    non_hosp.fillna(0, inplace=True)
    return hosp, non_hosp

# ── Load GLM results: from disk cache OR fit once then save ───────
@st.cache_data
def load_or_fit_glm_results():
    """
    Reads pre-saved CSVs from models/stat_results/.
    If any are missing, fits ALL models ONCE and saves them.
    Never re-fits on page reload.
    """
    os.makedirs(STAT_DIR, exist_ok=True)
    all_present = all(
        os.path.exists(os.path.join(STAT_DIR, f"{k}.csv")) for k in MODEL_KEYS
    )

    if all_present:
        return {k: pd.read_csv(os.path.join(STAT_DIR, f"{k}.csv")) for k in MODEL_KEYS}

    # Only reaches here on first ever run
    hosp, non_hosp = load_data()
    from statistics_models import run_all_models
    results = run_all_models(hosp, non_hosp)
    for k, df in results.items():
        df.to_csv(os.path.join(STAT_DIR, f"{k}.csv"), index=False)
    return results

try:
    load_data()   # validate data exists first
except FileNotFoundError:
    st.error("Data files not found. Run the data pipeline first.")
    st.stop()

cache_ready = all(
    os.path.exists(os.path.join(STAT_DIR, f"{k}.csv")) for k in MODEL_KEYS
)

if not cache_ready:
    st.info("🔄 First run — fitting statistical models and saving to disk. This takes ~1–2 min and won't happen again.")

with st.spinner("Loading statistical results..." if cache_ready else "Fitting models for the first time..."):
    try:
        results = load_or_fit_glm_results()
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        st.stop()

if cache_ready:
    st.success("✅ Results loaded from saved cache (no re-fitting).")

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Outcome", [
    "OOPE (Gamma GLM)",
    "CHE (Binomial GLM)",
    "Poverty Line (Binomial GLM)",
    "Distress (Binomial GLM)"
])
dataset_choice = st.sidebar.radio("Dataset", ["Hospitalization", "Non-Hospitalization"])

key_map = {
    ("OOPE (Gamma GLM)",            "Hospitalization"):     "oope_hosp",
    ("OOPE (Gamma GLM)",            "Non-Hospitalization"): "oope_non_hosp",
    ("CHE (Binomial GLM)",          "Hospitalization"):     "che_hosp",
    ("CHE (Binomial GLM)",          "Non-Hospitalization"): "che_non_hosp",
    ("Poverty Line (Binomial GLM)", "Hospitalization"):     "pl_hosp",
    ("Poverty Line (Binomial GLM)", "Non-Hospitalization"): "pl_non_hosp",
    ("Distress (Binomial GLM)",     "Hospitalization"):     "distress_hosp",
    ("Distress (Binomial GLM)",     "Non-Hospitalization"): "distress_non_hosp",
}

result_key = key_map[(model_choice, dataset_choice)]
df_result  = results[result_key].copy()

sig_only = st.sidebar.checkbox("Show significant only (p < 0.05)", value=False)
if sig_only:
    df_result = df_result[df_result["Significant"] == True]

# ── Button to force refit (if user updated data) ──────────────────
if st.sidebar.button("🔁 Refit Models (clear cache)"):
    import shutil
    shutil.rmtree(STAT_DIR, ignore_errors=True)
    st.cache_data.clear()
    st.rerun()

st.subheader(f"{model_choice} — {dataset_choice}")

is_gamma = "Gamma" in model_choice

display_cols = (
    ["Variable", "Coefficient", "Std_Error", "t_value", "p_value",
     "CI_Lower", "CI_Upper", "Percent_Change", "Significant"]
    if is_gamma else
    ["Variable", "Coefficient", "Std_Error", "z_value", "p_value",
     "Odds_Ratio", "OR_CI_Lower", "OR_CI_Upper", "Significant"]
)
display_cols = [c for c in display_cols if c in df_result.columns]

def highlight_sig(row):
    color = "#d4edda" if row.get("Significant", False) else ""
    return [f"background-color: {color}"] * len(row)

st.dataframe(
    df_result[display_cols].style.apply(highlight_sig, axis=1),
    use_container_width=True, height=500
)

csv = df_result[display_cols].to_csv(index=False).encode()
st.download_button("⬇️ Download Results CSV", data=csv,
                   file_name=f"{result_key}_results.csv", mime="text/csv")

# ── Forest plot ───────────────────────────────────────────────────
# st.subheader("Forest Plot")

plot_col = "Odds_Ratio" if not is_gamma else "Percent_Change"
ci_lo    = "OR_CI_Lower" if not is_gamma else "CI_Lower"
ci_hi    = "OR_CI_Upper" if not is_gamma else "CI_Upper"

# if all(c in df_result.columns for c in [plot_col, ci_lo, ci_hi]):
#     plot_df = df_result[df_result["Variable"] != "Intercept"].dropna(subset=[plot_col]).head(30)
#     fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))
#     y_pos  = range(len(plot_df))
#     colors = ["#c62828" if s else "#78909c" for s in plot_df["Significant"]]
#     ax.barh(list(y_pos), plot_df[plot_col],
#             xerr=[plot_df[plot_col] - plot_df[ci_lo], plot_df[ci_hi] - plot_df[plot_col]],
#             color=colors, alpha=0.8, height=0.6, capsize=3)
#     ref_line = 1.0 if not is_gamma else 0.0
#     ax.axvline(ref_line, color="black", linewidth=1, linestyle="--")
#     # ax.set_yticks(list(y_pos)); ax.set_yticklabels(plot_df["Variable"], fontsize=8)
#     # ax.set_xlabel("Odds Ratio" if not is_gamma else "% Change in OOPE")
#     # ax.set_title(f"{model_choice} — Forest Plot\n🔴 significant  ⚫ not significant")
#     plt.tight_layout(); st.pyplot(fig); plt.close()

with st.expander("ℹ️ How to read this table"):
    if is_gamma:
        st.markdown("""
**Gamma GLM (log link) — OOPE**
- **Coefficient**: log-scale effect on OOPE
- **% Change**: `(exp(coeff) - 1) × 100` — percentage change in expected OOPE
- **p-value < 0.05** → statistically significant (highlighted green)

*Example: Coefficient = 0.35 → OOPE is ~42% higher for this group*
        """)
    else:
        st.markdown("""
**Binomial GLM (logit) — Binary outcomes**
- **Odds Ratio (OR)**: `exp(coefficient)` — OR > 1 = higher odds, OR < 1 = lower odds
- **CI**: 95% confidence interval
- **p-value < 0.05** → statistically significant (highlighted green)

*Example: OR = 2.5 → 2.5× higher odds of CHE for this group*
        """)