"""
pages/1_EDA.py — Exploratory Data Analysis
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from label_maps import NSS_STATE_MAP, DISEASE_COLS

st.set_page_config(page_title="EDA | NSSO", layout="wide")
st.title("📊 Exploratory Data Analysis")

# ── Load data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), "..", "data/processed_data")
    hosp     = pd.read_csv(os.path.join(base, "hospital_model.csv"))
    non_hosp = pd.read_csv(os.path.join(base, "non_hospital_model.csv"))
    hosp.fillna(0, inplace=True)
    non_hosp.fillna(0, inplace=True)
    hosp["state_name"]     = hosp["state"].map(NSS_STATE_MAP)
    non_hosp["state_name"] = non_hosp["state"].map(NSS_STATE_MAP)
    return hosp, non_hosp

try:
    hosp, non_hosp = load_data()
except FileNotFoundError:
    st.error("Data files not found in `data/`. Please run the data pipeline first.")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────
st.sidebar.header("Filters")
sector_filter = st.sidebar.selectbox("Sector", ["All", "Rural (1)", "Urban (2)"])

def apply_sector(df):
    if sector_filter == "Rural (1)":    return df[df["sector"] == 1]
    if sector_filter == "Urban (2)":    return df[df["sector"] == 2]
    return df

h  = apply_sector(hosp)
nh = apply_sector(non_hosp)

# ── Summary KPIs ──────────────────────────────────────────────────
st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)

def wt_pct(df, col):
    return round((df[col] * df["mult1"]).sum() / df["mult1"].sum() * 100, 1)

c1.metric("Hospitalized HHs",       f"{len(h):,}")
c2.metric("Outpatient HHs",         f"{len(nh):,}")
c3.metric("CHE % (Hosp)",           f"{wt_pct(h,  'CHE')} %")
c4.metric("CHE % (Non-Hosp)",       f"{wt_pct(nh, 'CHE')} %")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Distress % (Hosp)",      f"{wt_pct(h,  'distress')} %")
c6.metric("Distress % (Non-Hosp)",  f"{wt_pct(nh, 'distress')} %")
c7.metric("Poverty Push % (Hosp)",  f"{wt_pct(h,  'under_poverty_since_OOPE')} %")
c8.metric("Poverty Push % (NH)",    f"{wt_pct(nh, 'under_poverty_since_OOPE')} %")

st.markdown("---")

# ── Tab layout ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👥 Demographics", "💰 OOPE Distribution",
    "🗺️ State Analysis", "🏥 Hospital Type", "🦠 Disease"
])

# ─────────────────────────────────────────────────────────────────
# TAB 1: Demographics
# ─────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Demographic Distribution")

    col_map = {
        # "Gender":           {1: "Male", 2: "Female", 3: "Transgender"},
        "social_group":     {1: "ST", 2: "SC", 3: "OBC", 4: "General"},
        "religion":         {1: "Hindu", 2: "Islam", 3: "Christian", 4: "Sikh", 5: "Other"},
        "economic_quintile":{1: "Q1:9000", 2: "Q2:12500", 3: "Q3:16500", 4: "Q4:22000", 5: "Q5:above"},
    }

    chosen = st.selectbox("Select variable", list(col_map.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, df, title, color in zip(
        axes, [h, nh],
        ["Hospitalization", "Non-Hospitalization"],
        ["#1a7abf", "#e05c2a"]
    ):
        temp = df.copy()
        temp[chosen] = temp[chosen].map(col_map[chosen]).fillna("Unknown")
        counts = temp.groupby(chosen)["mult1"].sum().reset_index()
        counts.columns = [chosen, "weight"]
        counts["pct"] = counts["weight"] / counts["weight"].sum() * 100
        sns.barplot(data=counts, x=chosen, y="pct", ax=ax, color=color)
        ax.set_title(title)
        ax.set_ylabel("Weighted %")
        ax.set_xlabel(chosen.replace("_", " ").title())
        for bar in ax.patches:
            ax.annotate(f"{bar.get_height():.1f}%",
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # # Age distribution
    # st.subheader("Age Distribution (vs CHE cases)")
    # fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # for ax, df, title in zip([ax1, ax2], [h, nh], ["Hospitalization", "Non-Hospitalization"]):
    #     sns.histplot(df["Age"], kde=True, bins=50, color="#aac8e8", ax=ax, label="All")
    #     sns.histplot(df[df["CHE"] == 1]["Age"], kde=True, bins=50, color="#e05c2a", ax=ax, label="CHE=1")
    #     ax.set_title(title)
    #     ax.legend()
    # plt.tight_layout()
    # st.pyplot(fig2)
    # plt.close()

# ─────────────────────────────────────────────────────────────────
# TAB 2: OOPE Distribution
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("OOPE Distribution")

    log_transform = st.checkbox("Apply log transformation", value=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row, (df, label) in enumerate([(h, "Hospitalization"), (nh, "Non-Hospitalization")]):
        oope = np.log1p(df["OOPE"]) if log_transform else df["OOPE"]
        title_suffix = " (log scale)" if log_transform else ""

        sns.histplot(oope, kde=True, bins=80, ax=axes[row, 0], color="#1a7abf")
        axes[row, 0].set_title(f"{label} — OOPE Histogram{title_suffix}")
        axes[row, 0].set_xlabel("OOPE (Rs.)")

        sns.boxenplot(oope, ax=axes[row, 1], color="#1a7abf")
        axes[row, 1].set_title(f"{label} — OOPE Boxen Plot{title_suffix}")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # OOPE by income quintile
    st.subheader("Weighted Mean OOPE by Economic Quintile")
    q_map = {1: "Q1 :9000", 2: "Q2:12500", 3: "Q3:16500", 4: "Q4:22000", 5: "Q5:above"}

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, df, title, color in zip(
        [ax1, ax2], [h, nh],
        ["Hospitalization", "Non-Hospitalization"],
        ["#1a7abf", "#e05c2a"]
    ):
        wm = df.groupby("economic_quintile").apply(
            lambda x: (x["OOPE"] * x["mult1"]).sum() / x["mult1"].sum()
        ).reset_index()
        wm.columns = ["quintile", "weighted_oope"]
        wm["quintile"] = wm["quintile"].map(q_map)
        sns.barplot(data=wm, x="quintile", y="weighted_oope", ax=ax, color=color)
        ax.set_title(title)
        ax.set_ylabel("Weighted Mean OOPE (Rs.)")
        ax.set_xlabel("Economic Quintile")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ─────────────────────────────────────────────────────────────────
# TAB 3: State Analysis
# ─────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("State-wise Weighted Analysis")

    metric = st.selectbox("Metric", ["OOPE", "CHE", "distress", "under_poverty_since_OOPE"])

    def state_weighted(df, col):
        return df.groupby("state_name").apply(
            lambda x: (x[col] * x["mult1"]).sum() / x["mult1"].sum()
        ).reset_index(name="value").sort_values("value", ascending=True)

    sw_h  = state_weighted(h,  metric)
    sw_nh = state_weighted(nh, metric)

    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
    for ax, df, title, color in zip(
        [ax1, ax2], [sw_h, sw_nh],
        ["Hospitalization", "Non-Hospitalization"],
        ["#1a7abf", "#e05c2a"]
    ):
        ax.barh(df["state_name"], df["value"], color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_title(f"{title} — {metric}")
        ax.set_xlabel(metric)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

# ─────────────────────────────────────────────────────────────────
# TAB 4: Hospital Type
# ─────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Distress Financing by Hospital Type")

    outcome = st.selectbox("Outcome", ["distress", "CHE", "under_poverty_since_OOPE"], key="ht_out")

    def hosp_type_analysis(df, col):
        result = {}
        for ht in ["Govt", "Charity", "Private"]:
            if ht in df.columns:
                sub = df[df[ht] == 1]
                if len(sub) > 0:
                    result[ht] = (sub[col] * sub["mult1"]).sum() / sub["mult1"].sum() * 100
                else:
                    result[ht] = 0.0
        return pd.Series(result)

    r_h  = hosp_type_analysis(h,  outcome)
    r_nh = hosp_type_analysis(nh, outcome)

    fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    r_h.plot(kind="bar",  ax=ax1, color=["#2e7d32", "#f57c00", "#c62828"], rot=0)
    r_nh.plot(kind="bar", ax=ax2, color=["#2e7d32", "#f57c00", "#c62828"], rot=0)
    ax1.set_title(f"Hospitalization — {outcome} % by Hospital Type")
    ax2.set_title(f"Non-Hospitalization — {outcome} % by Hospital Type")
    ax1.set_ylabel("Weighted %")
    ax2.set_ylabel("Weighted %")
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

# ─────────────────────────────────────────────────────────────────
# TAB 5: Disease
# ─────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Outcome by Disease Category")

    disease_outcome = st.selectbox(
        "Outcome", ["CHE", "distress", "under_poverty_since_OOPE"], key="dis_out"
    )

    def disease_analysis(df, col):
        res = {}
        for d in DISEASE_COLS:
            if d in df.columns:
                sub = df[df[d] == 1]
                if len(sub) > 0:
                    res[d] = (sub[col] * sub["mult1"]).sum() / sub["mult1"].sum() * 100
        return pd.Series(res).sort_values()

    dh  = disease_analysis(h,  disease_outcome)
    dnh = disease_analysis(nh, disease_outcome)

    fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    dh.plot(kind="barh",  ax=ax1, color="#1a7abf")
    dnh.plot(kind="barh", ax=ax2, color="#e05c2a")
    ax1.set_title(f"Hospitalization — {disease_outcome} % by Disease")
    ax2.set_title(f"Non-Hospitalization — {disease_outcome} % by Disease")
    ax1.set_xlabel("Weighted %")
    ax2.set_xlabel("Weighted %")
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()
