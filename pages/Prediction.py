"""
pages/3_Prediction.py — ML Prediction Interface

FIX: Models loaded ONCE with @st.cache_resource — never retrained.
     Feature importance CSVs read from disk (saved by train_models.py).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from label_maps import NSS_STATE_MAP, DISEASE_COLS, HOSPITAL_TYPE_COLS

st.set_page_config(page_title="Prediction | NSSO", layout="wide")
st.title("🤖 ML Prediction — Household Risk Assessment")

st.warning(
    """
    ⚠️ Disclaimer
    The results provided by this dashboard are based on statistical and machine learning models developed using data from the NSSO 80th Round Health Consumption Expenditure Survey (2023–24).
    These predictions should be considered indicative estimates only and are intended to support understanding and decision-making.
    Since healthcare costs, policies, and economic conditions may change over time, actual outcomes may differ.
    Please use these results as a reference or suggestion rather than a guaranteed prediction.
    """
)

st.markdown("""
Enter household characteristics to predict the probability of:
- **CHE** — Catastrophic Health Expenditure
- **Poverty Push** — Pushed below poverty line due to OOPE
- **Distress** — Distress Financing (borrowed / sold assets for healthcare)
*Models trained once with XGBoost — loaded from disk on every page visit.*
""")
st.info("""
ℹ️ Please enter household characteristics corresponding to the healthcare event being analysed.
• Hospitalization cases refer to treatment requiring admission during the previous 365 days.
• Outpatient cases refer to treatment received without admission during the previous 15 days.
• Predictions are based on NSSO 80th Round (2025) survey data and should be treated as indicative estimates only.
""")

BASE       = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE, "models")

# ── Load saved model ONCE — never retrain ─────────────────────────
@st.cache_resource
def load_all_models():
    """Load all 6 pkl models into memory once. Cached for the session."""
    from ml_models import load_model
    keys = [
        ("CHE",                     "hosp"),
        ("CHE",                     "non_hosp"),
        ("under_poverty_since_OOPE","hosp"),
        ("under_poverty_since_OOPE","non_hosp"),
        ("distress",                "hosp"),
        ("distress",                "non_hosp"),
    ]
    models = {}
    for target, ds in keys:
        try:
            models[(target, ds)] = load_model(target, ds)
        except FileNotFoundError:
            models[(target, ds)] = None
    return models

# Check at least one model exists before proceeding
sample_path = os.path.join(MODELS_DIR, "xgb_CHE_hosp.pkl")
if not os.path.exists(sample_path):
    st.warning("⚠️ Models not found. Please run `python train_models.py` first, then reload this page.")
    st.stop()

models = load_all_models()
st.sidebar.success("✅ Models loaded from disk")

# ── Input form ────────────────────────────────────────────────────
st.subheader("Enter Household Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Household Characteristics**")
        sector     = st.selectbox("Sector",          ["Rural", "Urban"])
        hh_size    = st.selectbox("Household Size",  ["≤4 members", ">4 members"])
        hh_edu     = st.selectbox("Head Education",  ["Below Secondary", "Below Graduation", "Graduation & above"])
        religion   = st.selectbox("Religion",        ["Hinduism", "Islam", "Other"])
        social_grp = st.selectbox("Social Group",    ["ST", "SC", "OBC", "General"])
        income_src = st.selectbox("Income Source",   ["Self-employed", "Regular Wage", "Casual Labour", "Other"])

    with col2:
        st.markdown("**Household Head**")
        gender     = st.selectbox("Gender",          ["Male", "Female", "Transgender"])
        age_60     = st.selectbox("Head Age",        ["Below 60", "60 & above"])
        econ_q     = st.selectbox("Economic Quintile", [
            "1st Quintile (Poorest)", "2nd Quintile",
            "3rd Quintile", "4th Quintile", "5th Quintile (Richest)"
        ])
        insurance  = st.selectbox("Insurance Coverage", ["No", "Yes"])
        state_name = st.selectbox("State", sorted(NSS_STATE_MAP.values()))

        education_options = {
            "Not Literate": 1, "Non-formal Education": 2, "Below Primary": 3,
            "Primary": 4, "Upper Primary / Middle": 5, "Secondary": 6,
            "Higher Secondary": 7, "Diploma (Up to Secondary)": 8,
            "Diploma (Higher Secondary)": 10, "Diploma (Graduation & Above)": 11,
            "Graduate": 12, "Post Graduate & Above": 13
        }
        max_edu_label = st.selectbox(
             "Highest Education Level in Household",
             list(education_options.keys()),
             help="Highest educational attainment among all household members."
         )
        max_education = education_options[max_edu_label]

        female_ratio = st.slider(
              "Female Ratio in Household",
              0.0, 1.0, 0.50, 0.01,
              help="Proportion of household members who are female. Example: 0.50 means 50% of members are female."
          )

    with col3:
        st.markdown("**Healthcare Utilisation**")
        dataset_type = st.radio("Case Type",
                   ["Hospitalization", "Outpatient (15-day)"],
                   help="""
          Hospitalization: Medical conditions requiring admission during the last 365 days.
          Outpatient: Medical conditions treated without admission during the last 15 days.
          """)
        st.markdown("""
          *Select the type(s) of healthcare facility used for treatment.*
          
          - Hospitalization: Facility used during the last 365 days.
          - Outpatient: Facility used during the last 15 days.
          """)

        use_govt    = st.checkbox("Government Hospital")
        use_charity = st.checkbox("Charity Hospital")
        use_private = st.checkbox("Private Hospital")
        st.markdown("**Disease (select all that apply)**")
        selected_diseases = st.multiselect("Diseases",DISEASE_COLS,
                   help="""
               Select all diseases/ailments associated with the healthcare event.
               
               For Hospitalization cases:
               Diseases reported during the previous 365 days.
               
               For Outpatient cases:
               Diseases reported during the previous 15 days.
               """
               )

    submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

# ── Prediction (uses cached models — no retraining) ───────────────
if submitted:
    input_dict = {
        "sector":                   sector,
        "HH_size":                  hh_size,
        "HH_education":             hh_edu,
        "Religion":                 religion,
        "social_group":             social_grp,
        "Mejor_source_of_income":   income_src,
        "Gender":                   gender,
        "age_of_household_base60":  age_60,
        "economic_quintile":        econ_q,
        "insurance_cover":          1 if insurance == "Yes" else 0,
        "state":                    state_name,
        "Govt":                     int(use_govt),
        "Charity":                  int(use_charity),
        "Private":                  int(use_private),
        "max_education":            max_education,
        "female_ratio":             female_ratio,
    }
    for d in DISEASE_COLS:
        input_dict[d] = 1 if d in selected_diseases else 0

    ds_key = "hosp" if dataset_type == "Hospitalization" else "non_hosp"
    row    = pd.DataFrame([input_dict])

    st.markdown("---")
    st.subheader("Prediction Results")

    target_info = {
        "CHE":                      ("🔴", "Catastrophic Health Expenditure"),
        "under_poverty_since_OOPE": ("🟠", "Pushed into Poverty by OOPE"),
        "distress":                 ("🟡", "Distress Financing"),
    }

    all_probs   = {}
    result_cols = st.columns(3)

    for i, (target, (icon, label)) in enumerate(target_info.items()):
        with result_cols[i]:
            pipe = models.get((target, ds_key))
            if pipe is None:
                st.error(f"Model for {target}/{ds_key} not found.")
                continue
            try:
                prob = float(pipe.predict_proba(row)[0, 1]) * 100
                all_probs[label] = prob
                risk  = "High Risk 🔴" if prob > 60 else ("Moderate ⚠️" if prob > 35 else "Low Risk ✅")
                color = "#c62828"      if prob > 60 else ("#f57c00"      if prob > 35 else "#2e7d32")

                st.markdown(f"### {icon} {label}")
                st.metric("Probability", f"{prob:.1f}%")
                st.markdown(f"**{risk}**")

                fig, ax = plt.subplots(figsize=(3, 0.5))
                ax.barh([0], [prob],       color=color,    height=0.6)
                ax.barh([0], [100 - prob], left=[prob],    color="#eeeeee", height=0.6)
                ax.set_xlim(0, 100); ax.set_yticks([])
                ax.set_xticks([0, 50, 100])
                ax.set_xticklabels(["0%", "50%", "100%"], fontsize=7)
                for spine in ax.spines.values(): spine.set_visible(False)
                plt.tight_layout(pad=0.1)
                st.pyplot(fig, use_container_width=True); plt.close()
            except Exception as e:
                st.error(f"Prediction error: {e}")

    if all_probs:
        st.markdown("---")
        st.subheader("Risk Summary")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        labels = list(all_probs.keys()); values = list(all_probs.values())
        colors = ["#c62828" if v > 60 else ("#f57c00" if v > 35 else "#2e7d32") for v in values]
        bars   = ax2.barh(labels, values, color=colors, height=0.5)
        ax2.axvline(50, color="gray", linestyle="--", linewidth=1)
        ax2.set_xlim(0, 100); ax2.set_xlabel("Predicted Probability (%)")
        for bar, val in zip(bars, values):
            ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=10)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    with st.expander("📋 View Input Summary"):
        st.dataframe(pd.DataFrame([input_dict]).T.rename(columns={0: "Value"}), use_container_width=True)

# ── Model metrics ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Model Performance")

metrics_path = os.path.join(MODELS_DIR, "metrics_summary.csv")
if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path, index_col=0)
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)
else:
    st.info("Run `python train_models.py` to see model metrics here.")

# ── Feature Importance ────────────────────────────────────────────
st.markdown("---")
st.subheader("Feature Importance")

fi_model = st.selectbox("Select Model", [
    "xgb_CHE_hosp", "xgb_CHE_non_hosp",
    "xgb_distress_hosp", "xgb_distress_non_hosp",
    "xgb_under_poverty_since_OOPE_hosp",
    "xgb_under_poverty_since_OOPE_non_hosp"
])

fi_path = os.path.join(MODELS_DIR, f"feature_importance_{fi_model}.csv")
if os.path.exists(fi_path):
    fi_df = pd.read_csv(fi_path)
    st.bar_chart(fi_df.head(20).set_index("feature"))
    st.dataframe(fi_df.head(20), use_container_width=True)
else:
    st.info("Feature importance files not found. Run `python train_models.py` to generate them.")

with st.expander("ℹ️ About the Models"):
    st.markdown("""
**Algorithm:** XGBoost Classifier — trained once, saved to `models/`, loaded from disk on every visit.

**Features:** Sector, Gender, Age, Religion, Social group, Education, Economic quintile,
Income source, Hospital type, Insurance, Disease category, State.

**Training:** 80/20 stratified split · 5-fold stratified CV · ROC-AUC ≈ 0.85
    """)
