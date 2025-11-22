import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="BaaS AML Risk Monitoring",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .team-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title(" Navigation")

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = " Home"

# Navigation buttons
if st.sidebar.button(" Home", use_container_width=True):
    st.session_state.current_page = " Home"

if st.sidebar.button(" Dashboard", use_container_width=True):
    st.session_state.current_page = " Dashboard"

if st.sidebar.button(" Model Performance", use_container_width=True):
    st.session_state.current_page = " Model Performance"

if st.sidebar.button("Transaction Monitor", use_container_width=True):
    st.session_state.current_page = " Transaction Monitor"

if st.sidebar.button("Analytics", use_container_width=True):
    st.session_state.current_page = "Analytics"

page = st.session_state.current_page

st.sidebar.markdown("---")
st.sidebar.info(
    "**BaaS Risk Monitoring System**\n\n"
    "Powered by IBM Synthetic Data\n\n"
    "CUA MDA Capstone Project"
)

# ============================================================================
# HOME PAGE
# ============================================================================
if page == " Home":
    # Header
    st.markdown('<div class="main-header"> BaaS Anti-Money Laundering Risk Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Financial Crime Detection System</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project Overview Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("##  Project Overview")
        st.markdown("""
        This dashboard presents a **machine learning-powered Anti-Money Laundering (AML) detection system** 
        built using IBM's Synthetic Core Banking and Money Laundering datasets.
        
        ### Objectives
        - **Detect** high-risk money laundering transactions in real-time
        - **Reduce** false positives to minimize investigation costs
        - **Improve** precision and recall for better F1 scores
        - **Provide** actionable insights for compliance teams
        
        ###  Dataset Characteristics
        - **31.9 Million** transactions analyzed
        - **35,230** labeled laundering cases (0.11%)
        - **28 days** of synthetic banking data (Sept 2022)
        - **Multiple currencies** and payment formats
        """)
        
        st.markdown("###  Key Features")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            -  Advanced feature engineering
            -  Network graph analysis
            -  Temporal pattern detection
            -  Velocity-based anomalies
            """)
        with col_b:
            st.markdown("""
            -  Ensemble ML models
            -  Real-time risk scoring
            -  Interactive visualizations
            -  Explainable AI insights
            """)
    
    with col2:
        st.markdown("##  Current Model Performance")
        
        # Performance metrics (update these with your actual values)
        metrics_data = {
            "Metric": ["Precision", "Recall", "F1 Score", "ROC-AUC"],
            "Score": [78.47, 53.50, 63.62, 98.78]
        }
        
        for metric, score in zip(metrics_data["Metric"], metrics_data["Score"]):
            st.metric(label=metric, value=f"{score}%")
        
        st.markdown("---")
        st.markdown("###  Target Goals")
        st.progress(0.85, text="Precision: 85%+")
        st.progress(0.75, text="Recall: 75%+")
        st.progress(0.80, text="F1 Score: 80%+")
    
    st.markdown("---")
    
    # Technical Approach
    st.markdown("## Technical Approach")
    
    tab1, tab2, tab3, tab4 = st.tabs([" Data Processing", "ML Models", " Class Imbalance", "Evaluation"])
    
    with tab1:
        st.markdown("""
        ### Data Processing Pipeline
        
        1. **Data Cleaning**
           - Removed 20 duplicate transactions
           - Validated amount consistency across currencies
           - Converted data types for memory optimization (2.6GB → 1.5GB)
        
        2. **Feature Engineering**
           - Temporal features (hour, day_of_week, weekend)
           - Transaction velocity (24h count, time since last)
           - Rolling averages (7-day, 30-day patterns)
           - Amount ratios and z-scores
        
        3. **Data Leakage Prevention**
           - Removed 12,458 future transactions from known criminals
           - Time-based train/test split (weeks 35-36 train, 37-39 test)
        """)
    
    with tab2:
        st.markdown("""
        ### Machine Learning Models
        
        **Ensemble Approach:**
        -  **XGBoost**: Gradient boosting with scale_pos_weight
        -  **LightGBM**: Fast training on large datasets
        -  **CatBoost**: Handles categorical features natively
        -  **Random Forest**: Baseline comparison
        
        **Optimization Techniques:**
        - Hyperparameter tuning via early stopping
        - Threshold optimization for F1 maximization
        - Feature importance analysis (gain & permutation)
        - Model calibration (isotonic regression)
        """)
    
    with tab3:
        st.markdown("""
        ### Handling Class Imbalance
        
        **Challenge**: 99.89% normal vs 0.11% laundering transactions
        
        **Solutions Implemented:**
        1. **Class Weighting**: scale_pos_weight = 1,230.94
        2. **SMOTE**: Synthetic minority oversampling (on subset)
        3. **Undersampling**: Random majority class reduction
        4. **Hybrid**: Combined SMOTE + undersampling
        5. **Cost-Sensitive Learning**: Custom loss functions
        
        **Best Strategy**: Class weighting + threshold tuning (0.987)
        """)
    
    with tab4:
        st.markdown("""
        ### Evaluation Metrics
        
        **Primary Metrics:**
        - **Precision**: Minimize false alarms (investigation cost)
        - **Recall**: Catch actual laundering cases (regulatory risk)
        - **F1 Score**: Harmonic mean for balanced performance
        - **PR-AUC**: Better than ROC-AUC for imbalanced data
        
        **Business Impact:**
        - Each false positive costs ~$500 in investigation time
        - Each missed case costs ~$50,000 in fines/reputation
        - Target: 80%+ F1 score for production deployment
        """)
    
    st.markdown("---")
    
    # Team Section
    st.markdown("## Team Members")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="team-card">
            <h3> Delphin Kaduli</h3>
        </div>
        """, unsafe_allow_html=True)    
    
    with col2:
        st.markdown("""
        <div class="team-card">
            <h3> Tycho Janssen</h3
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="team-card">
            <h3> Solomon Pinto</h3
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tech Stack
    st.markdown("## Technology Stack")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **Data Processing**
        - Pandas
        - NumPy
        - PyArrow
        - S3FS
        """)
    
    with col2:
        st.markdown("""
        **Machine Learning**
        - XGBoost
        - LightGBM
        - CatBoost
        - Scikit-learn
        - Imbalanced-learn
        """)
    
    with col3:
        st.markdown("""
        **Visualization**
        - Streamlit
        - Plotly
        - Seaborn
        - Matplotlib
        """)
    
    with col4:
        st.markdown("""
        **Infrastructure**
        - AWS S3
        - GitHub
        - Jupyter
        - Python 3.11
        """)
    
    st.markdown("---")
    
    # Call to Action
    st.markdown("## Get Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(" View Dashboard", use_container_width=True):
            st.info("Navigate using the sidebar to explore the dashboard")
    
    with col2:
        if st.button(" Model Performance", use_container_width=True):
            st.info("Check detailed model metrics and comparisons")
    
    with col3:
        if st.button("Monitor Transactions", use_container_width=True):
            st.info("Real-time transaction risk assessment")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 2rem 0;">
        <p><strong>CUA MDA Capstone Project 2025</strong></p>
        <p>Built with ❤️ using IBM Synthetic Banking Data</p>
        <p>Contact: <a href="https://github.com/DelphinKdl/CUA-MDA-Capstone-BaaS-Risk-Monitoring">GitHub Repository</a></p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PLACEHOLDER PAGES (To be implemented)
# ============================================================================
elif page == " Dashboard":
    st.title(" AML Monitoring Dashboard")
    st.info("Dashboard page under construction. This will show real-time transaction monitoring.")
    st.markdown("""
    ### Coming Soon:
    - Real-time transaction feed
    - Risk score distribution
    - Alert management system
    - Geographic transaction patterns
    """)

elif page == " Model Performance":
    st.title(" Model Performance Analysis")
    st.info("Model performance page under construction. This will show detailed metrics and comparisons.")
    st.markdown("""
    ### Coming Soon:
    - Confusion matrix visualization
    - ROC and PR curves
    - Feature importance rankings
    - Model comparison tables
    - Threshold optimization charts
    """)

elif page == "Transaction Monitor":
    st.title("Transaction Risk Monitor")
    st.info("Transaction monitor under construction. This will allow searching and scoring individual transactions.")
    st.markdown("""
    ### Coming Soon:
    - Transaction search by ID
    - Real-time risk scoring
    - SHAP explanations
    - Similar transaction patterns
    - Investigation workflow
    """)

elif page == " Analytics":
    st.title(" Advanced Analytics")
    st.info(" Analytics page under construction. This will show deep-dive analysis and trends.")
    st.markdown("""
    ### Coming Soon:
    - Temporal trend analysis
    - Network graph visualization
    - Currency flow patterns
    - Anomaly detection insights
    - Custom report generation
    """)