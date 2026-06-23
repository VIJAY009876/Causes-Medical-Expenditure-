"""
app.py — NSSO Health Expenditure Analysis Dashboard
====================================================
Entry point. Run with:  streamlit run app.py
"""

import streamlit as st

# ── MUST be first Streamlit command ────────────────────────────────
st.set_page_config(
    page_title="Analysis and Prediction of Medical Expenditure and Its Effects on Indian households",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {background: #0f2a4a;}
    [data-testid="stSidebar"] * {color: #e8f4f8 !important;}

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f0f7ff;
        border: 1px solid #c8e0f4;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Headers */
    h1 {color: #0f2a4a; font-family: 'Georgia', serif;}
    h2, h3 {color: #1a4a7a;}

    /* Info boxes */
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #1a7abf;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

st.info(
    """
    ℹ️ Results and predictions are based on NSSO 80th Round survey data (2023–24) and should be treated as indicative estimates rather than guaranteed outcomes.
    """
)

# ── Home page ─────────────────────────────────────────────────────
st.title("🏥 Analysis and Prediction of Medical Expenditure and Its Effects on Indian households
st.markdown("**80th Round — National Sample Survey**")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    ### 📊 Exploratory Analysis
    Visualise OOPE, CHE, distress financing and poverty across
    states, sectors, income groups, diseases and hospital types.
    """)
    st.page_link("pages/EDA.py", label="Open EDA →", icon="📊")

with col2:
    st.markdown("""
    ### 📐 Statistical Models
    Weighted GLM results — Gamma for OOPE, Logistic for CHE,
    Poverty Line, and Distress. Odds ratios with significance.
    """)
    st.page_link("pages/Statistics.py", label="Open Statistics →", icon="📐")

with col3:
    st.markdown("""
    ### 🤖 ML Prediction
    Enter household characteristics and get predicted probability
    of CHE, poverty after OOPE, and distress financing.
    """)
    st.page_link("pages/Prediction.py", label="Open Prediction →", icon="🤖")

st.markdown("---")
st.markdown("""
<div class="info-box">
<b>About this Survey:</b> The NSSO 80th Round Health Consumption Expenditure Survey
covers hospitalization (Block 6,7) and outpatient care (Block 8,9) for households across
all Indian states. Analysis uses multiplier weights (<code>mult1</code>) for
population-representative estimates.
</div>
""", unsafe_allow_html=True)

st.caption("Data: NSSO 80th Round | Analysis: NSS Health CEA Project")
