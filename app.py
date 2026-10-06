"""
Streamlit Web Application:
Road Accident Data Analysis, Visualization and Accident Severity Prediction Using Machine Learning

Academic Capstone Dashboard — Polished Light Theme & Enhanced Layout Edition
Features:
- Expanded Sidebar with increased width and elevated font sizes for effortless navigation
- Harmonized typography, text, and graph alignment across all views
- Full Light Mode UI with domain-specific traffic safety palette
- Dynamic dataset exploration & KPIs
- Two prominent ML navigation buttons:
  1. 🤖 Logistic Regression
  2. 🌲 Random Forest Classifier
- Live inference forms with real model prediction and probability metrics
- Model performance comparison
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_cleaning import clean_dataset, inspect_data, detect_outliers_iqr
from src.data_transformation import transform_data, run_full_pipeline, assign_time_of_day
from src.data_analysis import (
    load_processed_data,
    get_descriptive_statistics,
    get_grouped_statistics,
    perform_wrangling_operations,
    answer_eda_questions
)
from src.visualization import (
    plot_severity_distribution,
    plot_accidents_by_weather,
    plot_accidents_by_road_surface,
    plot_accidents_by_hour,
    plot_casualty_distribution,
    plot_vehicles_vs_casualties_scatter,
    plot_box_casualties_by_severity,
    plot_correlation_heatmap,
    plot_day_of_week_countplot,
    plot_light_conditions_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison
)
from src.models import train_and_evaluate_models, predict_severity

# Configure page settings
st.set_page_config(
    page_title="Road Accident Severity Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enhanced CSS for Expanded Sidebar, Increased Fonts, and Symmetric Graph Alignment
st.markdown("""
<style>
    /* Global Light Theme Styling */
    .stApp {
        background-color: #ffffff;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* --------------------------------------------------
       1. STATIC SIDEBAR (HIGH VISIBILITY, NO CLIPPING)
       -------------------------------------------------- */
    section[data-testid="stSidebar"] {
        min-width: 310px !important;
        max-width: 330px !important;
        background-color: #f8fafc !important;
        border-right: 1.5px solid #e2e8f0 !important;
    }

    /* Remove excessive whitespace at the bottom of the page and fix container spacing */
    html, body {
        overflow-x: hidden;
    }

    .main {
        overflow-x: hidden;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 95% !important;
    }

    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        padding: 0px !important;
    }

    /* Prevent Streamlit from creating empty scrolling space at the bottom */
    [data-testid="stAppViewBlockContainer"] {
        padding-bottom: 1rem !important;
    }

    div[data-testid="stBottomBlockContainer"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
        padding-top: 0.8rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        overflow-y: hidden !important;
        overflow-x: hidden !important;
    }

    /* Clean scrollbar styling */
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"]::-webkit-scrollbar {
        width: 4px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"]::-webkit-scrollbar-thumb {
        background: #cbd5e1 !important;
        border-radius: 4px !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.98rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
        padding: 6px 10px !important;
        margin-bottom: 2px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: #e2e8f0 !important;
        color: #1e40af !important;
    }

    section[data-testid="stSidebar"] .sidebar-caption {
        font-size: 0.82rem !important;
        color: #64748b !important;
        line-height: 1.35 !important;
    }

    /* --------------------------------------------------
       2. HERO BANNER & HEADER ALIGNMENT
       -------------------------------------------------- */
    .hero-banner {
        background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 50%, #f0fdf4 100%);
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .main-header {
        font-size: 2.25rem;
        font-weight: 800;
        color: #1e3a8a;
        line-height: 1.25;
        margin-bottom: 8px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        font-weight: 450;
        line-height: 1.5;
        margin-bottom: 0;
    }
    
    /* --------------------------------------------------
       3. BALANCED KPI METRIC CARDS
       -------------------------------------------------- */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 115px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px -2px rgba(0,0,0,0.08);
    }
    .kpi-total { border-top: 5px solid #2563eb; }
    .kpi-slight { border-top: 5px solid #059669; }
    .kpi-serious { border-top: 5px solid #d97706; }
    .kpi-fatal { border-top: 5px solid #dc2626; }
    .kpi-casualties { border-top: 5px solid #7c3aed; }
    
    .kpi-title {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.95rem;
        font-weight: 800;
        line-height: 1;
        margin: 4px 0;
    }
    .kpi-subtitle {
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
    }

    /* --------------------------------------------------
       4. PROMINENT ACTION CARDS (PERFECTLY ALIGNED)
       -------------------------------------------------- */
    .action-card {
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        min-height: 195px;
        height: auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        box-sizing: border-box;
    }
    .action-card-audit {
        background-color: #f0f9ff;
        border: 2px solid #bae6fd;
    }
    .action-card-lr {
        background-color: #eff6ff;
        border: 2px solid #bfdbfe;
    }
    .action-card-rf {
        background-color: #f0fdf4;
        border: 2px solid #bbf7d0;
    }
    .action-icon {
        font-size: 2.2rem;
        line-height: 1;
        margin-bottom: 8px;
    }
    .action-title {
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .action-desc {
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.45;
        margin: 0;
    }

    /* --------------------------------------------------
       5. GRAPH & TEXT CARD WRAPPERS (SYMMETRIC ALIGNMENT)
       -------------------------------------------------- */
    .card-panel {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .section-title {
        color: #1e3a8a;
        font-weight: 750;
        font-size: 1.3rem;
        margin-bottom: 14px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
    }

    .graph-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* Severity Status Badges */
    .badge-slight {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-serious {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-fatal {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CACHED DATA & MODEL LOADING
# ==========================================

@st.cache_data
def get_dataset():
    data_path = "data/processed/road_accidents_cleaned.csv"
    if not os.path.exists(data_path):
        df = run_full_pipeline()
    else:
        df = pd.read_csv(data_path)
    return df


@st.cache_data
def get_raw_dataset():
    raw_path = "data/raw/road_accidents.csv"
    return pd.read_csv(raw_path)


@st.cache_resource
def get_model_artifacts():
    metrics_path = "models/evaluation_metrics.joblib"
    if not os.path.exists(metrics_path):
        results = train_and_evaluate_models()
    else:
        results = joblib.load(metrics_path)
    return results


# Initialize session state for navigation
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🏠 Home"
if "sidebar_radio" not in st.session_state:
    st.session_state.sidebar_radio = st.session_state.nav_page


def set_page(page_name):
    st.session_state.nav_page = page_name
    st.session_state.sidebar_radio = page_name


def on_radio_change():
    st.session_state.nav_page = st.session_state.sidebar_radio


# Load data and ML results
df = get_dataset()
ml_results = get_model_artifacts()

# ==========================================
# STATIC SIDEBAR (COMPACT & NO SCROLL)
# ==========================================
st.sidebar.markdown("""
<div style="text-align: center; padding: 4px 0 6px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 8px;">
    <div style="display: flex; align-items: center; justify-content: center; gap: 6px;">
        <span style="font-size: 1.6rem;">🚦</span>
        <h2 style="color: #1e3a8a; margin: 0; font-weight: 850; font-size: 1.25rem; display: inline;">Road Safety AI</h2>
    </div>
    <span style="background-color: #dbeafe; color: #1e40af; padding: 1px 8px; border-radius: 10px; font-size: 0.74rem; font-weight: 700; margin-top: 2px; display: inline-block;">Academic Capstone</span>
</div>
""", unsafe_allow_html=True)

pages = [
    "🏠 Home",
    "🔄 Raw vs Processed Data",
    "🤖 Logistic Regression",
    "🌲 Random Forest"
]

selected_page = st.sidebar.radio(
    "",
    pages,
    key="sidebar_radio",
    on_change=on_radio_change,
    label_visibility="collapsed"
)

# Keep session state in sync
st.session_state.nav_page = selected_page

st.sidebar.markdown("""
<div style="background-color: #ffffff; border: 1.5px solid #cbd5e1; border-radius: 10px; padding: 12px 14px; margin-top: 14px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
    <div style="font-size: 0.8rem; font-weight: 800; color: #1e40af; letter-spacing: 0.05em; margin-bottom: 8px; text-transform: uppercase;">Project Overview</div>
    <div style="font-size: 0.92rem; color: #0f172a; line-height: 1.65;">
        <p style="margin: 0 0 6px 0;">📘 <strong>Course:</strong> Data Analytics & Visualization</p>
        <p style="margin: 0 0 6px 0;">🚗 <strong>Domain:</strong> Highway Traffic Safety</p>
        <p style="margin: 0 0 6px 0;">🧠 <strong>Models:</strong> Logistic Regression & RF</p>
        <p style="margin: 0;">📊 <strong>Records:</strong> 12,316 Incidents</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# PAGE 1: HOME
# ==========================================
if st.session_state.nav_page == "🏠 Home":
    st.markdown("""
    <div class="hero-banner">
        <div class="main-header">Road Accident Data Analysis, Visualization & Severity Prediction</div>
        <div class="sub-header">An End-to-End Academic Pipeline: Ingestion → Cleaning → Wrangling → Statistical EDA → Visualizations → Supervised Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

    col_btn0, col_btn1, col_btn2 = st.columns(3, gap="medium")
    with col_btn0:
        st.markdown("""
        <div class="action-card action-card-audit">
            <div class="action-icon">🔄</div>
            <div class="action-title" style="color: #0369a1;">Raw vs Processed Data</div>
            <div class="action-desc">Inconsistent vs consistent values before & after cleaning.</div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Dataset", use_container_width=True, type="primary", on_click=set_page, args=("🔄 Raw vs Processed Data",))

    with col_btn1:
        st.markdown("""
        <div class="action-card action-card-lr">
            <div class="action-icon">🤖</div>
            <div class="action-title" style="color: #1d4ed8;">Model 1: Logistic Regression</div>
            <div class="action-desc">Multi-class multinomial baseline with balanced class weighting.</div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Logistic Regression", use_container_width=True, type="primary", on_click=set_page, args=("🤖 Logistic Regression",))

    with col_btn2:
        st.markdown("""
        <div class="action-card action-card-rf">
            <div class="action-icon">🌲</div>
            <div class="action-title" style="color: #15803d;">Model 2: Random Forest</div>
            <div class="action-desc">100-tree ensemble with Gini feature importances & 81.5% accuracy.</div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Random Forest", use_container_width=True, type="primary", on_click=set_page, args=("🌲 Random Forest",))

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # BALANCED TWO-COLUMN LAYOUT: EXECUTIVE SUMMARY & DONUT GRAPH
    col_desc, col_pie = st.columns([1.2, 1.0], gap="large")
    with col_desc:
        st.markdown("""
        <div class="card-panel" style="height: 100%;">
            <div class="section-title">Executive Summary & Pipeline Architecture</div>
            <p style="color: #334155; line-height: 1.7; font-size: 1rem;">
            This study investigates highway traffic accidents to classify crash severity into 
            <span class="badge-slight">Slight Injury</span>, 
            <span class="badge-serious">Serious Injury</span>, and 
            <span class="badge-fatal">Fatal Injury</span>. 
            The research leverages an authentic traffic collision dataset comprising <strong>12,316 real-world police records</strong> with <strong>32 raw attributes</strong> expanded to <strong>39 engineered features</strong>.
            </p>
            <ul style="color: #334155; line-height: 1.8; font-size: 0.98rem; padding-left: 22px;">
                <li><strong>Data Hygiene:</strong> Median imputation for continuous metrics; mode imputation for categorical attributes; preservation of multi-vehicle pile-up outliers.</li>
                <li><strong>Feature Derivation:</strong> Constructed Crash Hour, Time of Day (Morning/Afternoon/Evening/Night), Rush Hour windows, and Day Type (Weekday vs. Weekend).</li>
                <li><strong>Predictive Machine Learning:</strong> Exact empirical comparison between <strong>Multinomial Logistic Regression</strong> and <strong>Random Forest Classifier</strong> under balanced class constraints.</li>
                <li><strong>Interactive Deployment:</strong> Integrated inference engine supporting real-time severity risk assessment from user scenario inputs.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_pie:
        st.markdown("""
        <div class="graph-card">
            <div class="section-title" style="width: 100%; text-align: left;">Ground-Truth Severity Distribution</div>
        """, unsafe_allow_html=True)
        fig_sev = plot_severity_distribution(df)
        st.pyplot(fig_sev, use_container_width=True)
        plt.close(fig_sev)
        st.markdown("</div>", unsafe_allow_html=True)



# ==========================================
# PAGE 2: RAW VS PROCESSED DATA (BEFORE & AFTER)
# ==========================================
elif st.session_state.nav_page == "🔄 Raw vs Processed Data":
    st.markdown("""
    <div class="hero-banner" style="border-left: 6px solid #0284c7;">
        <div class="main-header" style="color: #0369a1;">Dataset Audit: Raw (Inconsistent) vs Processed (Consistent)</div>
        <div class="sub-header">Direct empirical evaluation of dataset quality, missing value imputation, character encoding repairs, and standardization before & after cleaning</div>
    </div>
    """, unsafe_allow_html=True)

    raw_df = get_raw_dataset()
    clean_df = df

    # VIEW TOGGLE BUTTONS
    st.markdown("### 🔀 Select Dataset View Mode")
    if "data_view_mode" not in st.session_state:
        st.session_state.data_view_mode = "Side-by-Side (Before & After)"

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("🔴 1. View Raw Dataset (Before - Inconsistent)", use_container_width=True, type="primary" if st.session_state.data_view_mode == "Raw" else "secondary"):
            st.session_state.data_view_mode = "Raw"
            st.rerun()
    with btn_col2:
        if st.button("🟢 2. View Processed Dataset (After - Consistent)", use_container_width=True, type="primary" if st.session_state.data_view_mode == "Processed" else "secondary"):
            st.session_state.data_view_mode = "Processed"
            st.rerun()
    with btn_col3:
        if st.button("⚖️ 3. Side-by-Side Comparison (Before vs After)", use_container_width=True, type="primary" if st.session_state.data_view_mode == "Side-by-Side (Before & After)" else "secondary"):
            st.session_state.data_view_mode = "Side-by-Side (Before & After)"
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


    # ACTIVE DATASET VIEWER BASED ON BUTTON SELECTION
    st.markdown("### 📊 Interactive Dataset Viewer")
    row_count = st.slider("Preview rows to display:", min_value=10, max_value=500, value=50, step=10)

    if st.session_state.data_view_mode == "Raw":
        st.markdown(f"""
        <div style="background-color: #fef2f2; border: 1.5px solid #fecaca; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
            <h4 style="color: #991b1b; margin: 0;">🔴 Currently Viewing: Raw Dataset (Before Cleaning)</h4>
            <p style="color: #7f1d1d; margin: 4px 0 0 0; font-size: 0.92rem;">Contains 20,057 missing values, unstandardized text labels, character artifacts ('?'), and 32 raw features.</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(raw_df.head(row_count), use_container_width=True)

    elif st.session_state.data_view_mode == "Processed":
        st.markdown(f"""
        <div style="background-color: #f0fdf4; border: 1.5px solid #bbf7d0; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
            <h4 style="color: #166534; margin: 0;">🟢 Currently Viewing: Processed Dataset (After Cleaning)</h4>
            <p style="color: #14532d; margin: 4px 0 0 0; font-size: 0.92rem;">100% clean and consistent: 0 missing values, normalized categories, repaired encoding, and 39 engineered features ready for ML.</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(clean_df.head(row_count), use_container_width=True)

    else:
        st.markdown(f"""
        <div style="background-color: #f0f9ff; border: 1.5px solid #bae6fd; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
            <h4 style="color: #0369a1; margin: 0;">⚖️ Currently Viewing: Side-by-Side Comparison (Before vs After)</h4>
            <p style="color: #075985; margin: 4px 0 0 0; font-size: 0.92rem;">Compare the raw inconsistent data against the processed consistent data side by side.</p>
        </div>
        """, unsafe_allow_html=True)
        col_side1, col_side2 = st.columns(2, gap="medium")
        with col_side1:
            st.markdown(f"**🔴 Raw Dataset (12,316 × {raw_df.shape[1]}) — First {row_count} Rows:**")
            st.dataframe(raw_df.head(row_count), use_container_width=True)
        with col_side2:
            st.markdown(f"**🟢 Processed Dataset (12,316 × {clean_df.shape[1]}) — First {row_count} Rows:**")
            st.dataframe(clean_df.head(row_count), use_container_width=True)
elif st.session_state.nav_page == "🤖 Logistic Regression":
    st.markdown("""
    <div class="hero-banner" style="border-left: 6px solid #2563eb;">
        <div class="main-header" style="color: #1d4ed8;">Model 1: Multinomial Logistic Regression</div>
        <div class="sub-header">Multi-class L2-regularized logistic classification with balanced class weighting</div>
    </div>
    """, unsafe_allow_html=True)

    lr_data = ml_results["logistic_regression"]

    # DYNAMIC PERFORMANCE METRIC CARDS
    st.markdown("### 📈 Dynamically Evaluated Test Holdout Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Test Accuracy", f"{lr_data['accuracy']*100:.2f}%")
    with m2:
        st.metric("Weighted Precision", f"{lr_data['precision']*100:.2f}%")
    with m3:
        st.metric("Weighted Recall", f"{lr_data['recall']*100:.2f}%")
    with m4:
        st.metric("Weighted F1 Score", f"{lr_data['f1_score']:.4f}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    col_cm, col_rep = st.columns([1.1, 1.0], gap="large")
    with col_cm:
        st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title' style='width: 100%;'>Confusion Matrix Heatmap</div>", unsafe_allow_html=True)
        fig_cm = plot_confusion_matrix(lr_data["confusion_matrix"], ml_results["class_names"], "Logistic Regression")
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_rep:
        st.markdown("<div class='card-panel' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Classification Report</div>", unsafe_allow_html=True)
        report_df = pd.DataFrame(lr_data["report_dict"]).T
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
        st.caption("Evaluated on 2,464 test samples using the serialized preprocessing pipeline.")
        st.markdown("</div>", unsafe_allow_html=True)

    # INTERACTIVE USER PREDICTION FORM
    st.markdown("---")
    st.markdown("### 🔮 Interactive Live Prediction: Logistic Regression")
    st.write("Configure traffic scenario parameters below to test the trained Logistic Regression model:")

    with st.form("lr_prediction_form"):
        p_col1, p_col2, p_col3 = st.columns(3, gap="medium")
        with p_col1:
            day = st.selectbox("Day of Week", sorted(df["Day_of_week"].unique()), index=0)
            driver_age = st.selectbox("Driver Age Band", sorted(df["Age_band_of_driver"].unique()), index=0)
            driver_sex = st.selectbox("Driver Sex", sorted(df["Sex_of_driver"].unique()), index=1)
            driving_exp = st.selectbox("Driving Experience", sorted(df["Driving_experience"].unique()), index=0)
            vehicle_type = st.selectbox("Vehicle Type", sorted(df["Type_of_vehicle"].unique()), index=0)

        with p_col2:
            weather = st.selectbox("Weather Condition", sorted(df["Weather_conditions"].unique()), index=1)
            road_surface = st.selectbox("Road Surface Condition", sorted(df["Road_surface_conditions"].unique()), index=0)
            light_cond = st.selectbox("Lighting Conditions", sorted(df["Light_conditions"].unique()), index=3)
            accident_area = st.selectbox("Accident Area", sorted(df["Area_accident_occured"].unique()), index=5)
            junction_type = st.selectbox("Junction Type", sorted(df["Types_of_Junction"].unique()), index=1)

        with p_col3:
            collision_type = st.selectbox("Collision Type", sorted(df["Type_of_collision"].unique()), index=5)
            movement = st.selectbox("Vehicle Movement", sorted(df["Vehicle_movement"].unique()), index=1)
            cause = st.selectbox("Cause of Accident", sorted(df["Cause_of_accident"].unique()), index=2)
            vehicles_num = st.number_input("Vehicles Involved", min_value=1, max_value=10, value=2, step=1)
            casualties_num = st.number_input("Casualties Incurred", min_value=1, max_value=10, value=1, step=1)
            hour_val = st.slider("Hour of Incident (0-23)", min_value=0, max_value=23, value=17)

        submit_lr = st.form_submit_button("🚨 PREDICT ACCIDENT SEVERITY", use_container_width=True, type="primary")

    if submit_lr:
        tod_val = assign_time_of_day(hour_val)
        input_data = {
            "Day_of_week": day,
            "Age_band_of_driver": driver_age,
            "Sex_of_driver": driver_sex,
            "Driving_experience": driving_exp,
            "Type_of_vehicle": vehicle_type,
            "Area_accident_occured": accident_area,
            "Types_of_Junction": junction_type,
            "Road_surface_conditions": road_surface,
            "Light_conditions": light_cond,
            "Weather_conditions": weather,
            "Type_of_collision": collision_type,
            "Vehicle_movement": movement,
            "Cause_of_accident": cause,
            "Time_of_day": tod_val,
            "Number_of_vehicles_involved": int(vehicles_num),
            "Number_of_casualties": int(casualties_num),
            "Hour": int(hour_val)
        }

        with st.spinner("Executing Logistic Regression inference..."):
            pred_out = predict_severity("logistic", input_data)

        badge_class = "badge-slight" if pred_out["prediction"] == "Slight Injury" else ("badge-serious" if pred_out["prediction"] == "Serious Injury" else "badge-fatal")
        st.markdown(f"""
        <div class="card-panel" style="margin-top: 16px;">
            <h3 style="margin: 0; color: #1e3a8a;">Prediction Result: <span class="{badge_class}">{pred_out['prediction']}</span></h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Model Class Probability Distribution:")
        prob_cols = st.columns(len(pred_out["probabilities"]))
        for idx, (c_name, p_val) in enumerate(pred_out["probabilities"].items()):
            with prob_cols[idx]:
                st.metric(label=c_name, value=f"{p_val*100:.1f}%")


# ==========================================
# PAGE 6: MODEL 2 — RANDOM FOREST
# ==========================================
elif st.session_state.nav_page == "🌲 Random Forest":
    st.markdown("""
    <div class="hero-banner" style="border-left: 6px solid #059669;">
        <div class="main-header" style="color: #15803d;">Model 2: Random Forest Classifier</div>
        <div class="sub-header">Ensemble Decision Forest (100 Trees) with balanced subsampling & Gini impurity splitting</div>
    </div>
    """, unsafe_allow_html=True)

    rf_data = ml_results["random_forest"]

    # DYNAMIC PERFORMANCE METRIC CARDS
    st.markdown("### 📈 Dynamically Evaluated Test Holdout Metrics")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Test Accuracy", f"{rf_data['accuracy']*100:.2f}%")
    with r2:
        st.metric("Weighted Precision", f"{rf_data['precision']*100:.2f}%")
    with r3:
        st.metric("Weighted Recall", f"{rf_data['recall']*100:.2f}%")
    with r4:
        st.metric("Weighted F1 Score", f"{rf_data['f1_score']:.4f}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    col_cm_rf, col_rep_rf = st.columns([1.1, 1.0], gap="large")
    with col_cm_rf:
        st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title' style='width: 100%;'>Confusion Matrix Heatmap</div>", unsafe_allow_html=True)
        fig_cm_rf = plot_confusion_matrix(rf_data["confusion_matrix"], ml_results["class_names"], "Random Forest")
        st.pyplot(fig_cm_rf, use_container_width=True)
        plt.close(fig_cm_rf)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_rep_rf:
        st.markdown("<div class='card-panel' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Classification Report</div>", unsafe_allow_html=True)
        report_df_rf = pd.DataFrame(rf_data["report_dict"]).T
        st.dataframe(report_df_rf.style.format("{:.4f}"), use_container_width=True)
        st.caption("Evaluated on 2,464 test samples using the serialized preprocessing pipeline.")
        st.markdown("</div>", unsafe_allow_html=True)

    # DYNAMIC FEATURE IMPORTANCE CHART
    st.markdown("---")
    st.markdown("### 🌲 Dynamic Feature Importance (Extracted from Trained Forest)")
    st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
    fig_feat = plot_feature_importance(rf_data["feature_importances"], ml_results["feature_names"], top_n=15)
    st.pyplot(fig_feat, use_container_width=True)
    plt.close(fig_feat)
    st.caption("Feature importances are computed dynamically from mean decrease in impurity across all 100 decision trees.")
    st.markdown("</div>", unsafe_allow_html=True)

    # INTERACTIVE USER PREDICTION FORM
    st.markdown("---")
    st.markdown("### 🔮 Interactive Live Prediction: Random Forest")
    st.write("Configure traffic scenario parameters below to test the trained Random Forest model:")

    with st.form("rf_prediction_form"):
        p_col1, p_col2, p_col3 = st.columns(3, gap="medium")
        with p_col1:
            day_rf = st.selectbox("Day of Week", sorted(df["Day_of_week"].unique()), index=0, key="rf_day")
            driver_age_rf = st.selectbox("Driver Age Band", sorted(df["Age_band_of_driver"].unique()), index=0, key="rf_age")
            driver_sex_rf = st.selectbox("Driver Sex", sorted(df["Sex_of_driver"].unique()), index=1, key="rf_sex")
            driving_exp_rf = st.selectbox("Driving Experience", sorted(df["Driving_experience"].unique()), index=0, key="rf_exp")
            vehicle_type_rf = st.selectbox("Vehicle Type", sorted(df["Type_of_vehicle"].unique()), index=0, key="rf_veh")

        with p_col2:
            weather_rf = st.selectbox("Weather Condition", sorted(df["Weather_conditions"].unique()), index=1, key="rf_w")
            road_surface_rf = st.selectbox("Road Surface Condition", sorted(df["Road_surface_conditions"].unique()), index=0, key="rf_r")
            light_cond_rf = st.selectbox("Lighting Conditions", sorted(df["Light_conditions"].unique()), index=3, key="rf_l")
            accident_area_rf = st.selectbox("Accident Area", sorted(df["Area_accident_occured"].unique()), index=5, key="rf_a")
            junction_type_rf = st.selectbox("Junction Type", sorted(df["Types_of_Junction"].unique()), index=1, key="rf_j")

        with p_col3:
            collision_type_rf = st.selectbox("Collision Type", sorted(df["Type_of_collision"].unique()), index=5, key="rf_c")
            movement_rf = st.selectbox("Vehicle Movement", sorted(df["Vehicle_movement"].unique()), index=1, key="rf_m")
            cause_rf = st.selectbox("Cause of Accident", sorted(df["Cause_of_accident"].unique()), index=2, key="rf_cause")
            vehicles_num_rf = st.number_input("Vehicles Involved", min_value=1, max_value=10, value=2, step=1, key="rf_v_num")
            casualties_num_rf = st.number_input("Casualties Incurred", min_value=1, max_value=10, value=1, step=1, key="rf_c_num")
            hour_val_rf = st.slider("Hour of Incident (0-23)", min_value=0, max_value=23, value=17, key="rf_h_num")

        submit_rf = st.form_submit_button("🚨 PREDICT ACCIDENT SEVERITY", use_container_width=True, type="primary")

    if submit_rf:
        tod_val_rf = assign_time_of_day(hour_val_rf)
        input_data_rf = {
            "Day_of_week": day_rf,
            "Age_band_of_driver": driver_age_rf,
            "Sex_of_driver": driver_sex_rf,
            "Driving_experience": driving_exp_rf,
            "Type_of_vehicle": vehicle_type_rf,
            "Area_accident_occured": accident_area_rf,
            "Types_of_Junction": junction_type_rf,
            "Road_surface_conditions": road_surface_rf,
            "Light_conditions": light_cond_rf,
            "Weather_conditions": weather_rf,
            "Type_of_collision": collision_type_rf,
            "Vehicle_movement": movement_rf,
            "Cause_of_accident": cause_rf,
            "Time_of_day": tod_val_rf,
            "Number_of_vehicles_involved": int(vehicles_num_rf),
            "Number_of_casualties": int(casualties_num_rf),
            "Hour": int(hour_val_rf)
        }

        with st.spinner("Executing Random Forest inference..."):
            pred_out_rf = predict_severity("random_forest", input_data_rf)

        badge_class_rf = "badge-slight" if pred_out_rf["prediction"] == "Slight Injury" else ("badge-serious" if pred_out_rf["prediction"] == "Serious Injury" else "badge-fatal")
        st.markdown(f"""
        <div class="card-panel" style="margin-top: 16px;">
            <h3 style="margin: 0; color: #15803d;">Prediction Result: <span class="{badge_class_rf}">{pred_out_rf['prediction']}</span></h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Model Class Probability Distribution:")
        prob_cols_rf = st.columns(len(pred_out_rf["probabilities"]))
        for idx, (c_name, p_val) in enumerate(pred_out_rf["probabilities"].items()):
            with prob_cols_rf[idx]:
                st.metric(label=c_name, value=f"{p_val*100:.1f}%")



