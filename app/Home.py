import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import joblib
import json
import os

# Page configuration
st.set_page_config(
    page_title="Dashboard Overview",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and configuration
@st.cache_resource
def load_model_artifacts():
    """Load trained model and related artifacts"""
    try:
        base_paths = ['../models/', './models/', '../../models/']
        model_path = None
        
        for base in base_paths:
            test_path = os.path.join(base, 'calibrated_lightgbm_model.pkl')
            if os.path.exists(test_path):
                model_path = base
                break
        
        if model_path is None:
            raise FileNotFoundError("Models directory not found")
        
        model = joblib.load(os.path.join(model_path, 'calibrated_lightgbm_model.pkl'))
        scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        
        config_path = os.path.join(model_path, 'model_config.json')
        features_path = os.path.join(model_path, 'feature_names.json')
        importance_path = os.path.join(model_path, 'feature_importance.json')
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        
        with open(importance_path, 'r') as f:
            feature_importance = pd.DataFrame(json.load(f))
        
        return model, scaler, config, feature_names, feature_importance
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None

model, scaler, model_config, feature_names, feature_importance = load_model_artifacts()

# CSS 
st.markdown("""
    <style>
    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .production-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .failed-label {
        color: #dc2626;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .nav-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


st.sidebar.markdown("---")

# Model Status
if model_config:
    st.sidebar.markdown("### Best Models")
    st.sidebar.success(f"**{model_config['model_name']}**")
    st.sidebar.markdown(f"**Status:** Production")
    st.sidebar.markdown(f"**Version:** {model_config['model_version']}")
    st.sidebar.markdown(f"**Threshold:** {model_config['optimal_threshold']}")
    st.sidebar.success(f"**LightGBM, coming SOON**")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Key Metrics")
    st.sidebar.markdown(f"**Alert Confidence:** {model_config['performance_metrics']['precision']*100:.2f}%")
    st.sidebar.markdown(f"**Case Capture:** {model_config['performance_metrics']['recall']*100:.2f}%")
    st.sidebar.markdown(f"**ROC-AUC:** {model_config['performance_metrics']['roc_auc']*100:.2f}%")
else:
    st.sidebar.warning("Model not loaded")

st.sidebar.markdown("---")
st.sidebar.caption("CUA MDA Capstone Project Fall'25")
st.sidebar.caption("Team: Delphin Kaduli, Tycho Janssen, Solomon Pinto")

# Main Page Header
st.markdown('<div class="main-header">Anti Money Laundering Detection System</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["Executive Summary", " Technical Deep Dive"])

# tab 1
with tab1:
    st.markdown("### Key Performance Indicators")
    
    if model_config:
        metrics = model_config['performance_metrics']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Alert Confidence (Precision)",
                value=f"{metrics['precision']*100:.2f}%",
                delta="14.5x vs Target",
                help="73% of flagged transactions are actual laundering cases"
            )
        
        with col2:
            st.metric(
                label="Case Capture Rate (Recall)",
                value=f"{metrics['recall']*100:.2f}%",
                delta="2,008 Confirmed Cases",
                help="Catches 19% of all laundering transactions"
            )
        
        with col3:
            st.metric(
                label="False Alarm Reduction",
                value="27x",
                delta="vs Baseline",
                help="Only 752 false positives out of 6.4M normal transactions"
            )
        
        with col4:
            st.metric(
                label="Daily Alert Volume",
                value="~860",
                delta="Manageable Workload",
                help="0.19% of daily transactions flagged for review"
            )
    
    st.markdown("---")
    
    # Champion Model - Business Focus
    st.markdown("### The best Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Why This Model Won**
        
        **LightGBM + Probability Calibration**
        - Decision Threshold: 0.10 (optimized)
        - Training: Full 25.5M transactions
        
        **Business Impact:**
        - Reduces analyst workload by 62% vs alternative models
        - 27x reduction in false positives
        - 73% confidence in every alert
        - Manageable 860 alerts per day
        
        **Key Advantage:**
        Probability calibration ensures reliable risk scores. 
        When the model says 10% risk, it means 10% risk.
        """)
    
    with col2:
        st.info("""
        **Risk Focus Strategy**
        
        **What We Target:**
        - High-value, clear-pattern laundering
        - ACH payment structuring
        - Weekend activity patterns
        - Threshold avoidance ($9K-$10K)
        
        **Trade-off:**
        - High precision (73%) over high recall (19%)
        - Catches 1 in 5 laundering cases, But every alert is highly reliable
        
        **Result:**
        Maximizes analyst efficiency and regulatory confidence.
        """)
    
    st.markdown("---")
    
    # What We Look For
    st.markdown("### What We Look For")
    st.markdown("*Quick reference for transaction review*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **High Risk Indicators:**
        
        1. **ACH Payment Format** → 49x baseline risk
           - 84% of all transactions use ACH
           - Enables rapid structuring and smurfing
        
        2. **Weekend Transactions** → 3x baseline risk
           - Reduced oversight and monitoring
           - Combined with ACH: 21.6x risk
        
        3. **UK Pound Structuring** → 8.3x baseline risk
           - Amounts in $9,000-$10,000 range
           - Avoiding $10K CTR reporting threshold
        
        4. **Bank 1004 Pattern**
           - Consistently associated with laundering
           - Possible money mule network
        """)
    
    with col2:
        st.markdown("""
        **Protective/Safe Indicators:**
        
        1. **Bank 800 Transactions**
           - Zero laundering cases detected
           - Completely clean record
           - Trusted institution
        
        2. **Night Transactions** → 0.53x baseline
           - Contrary to expectations
           - Automated legitimate payments
           - Actually safer than daytime
        
        3. **What Didn't Matter:**
           - Round amounts ($1,000, $100)
           - Bitcoin (NOT highest risk)
           - Cross-currency transactions
        """)
    
    st.markdown("---")
    
    # Top 5 Features Chart
    st.markdown("### Top 5 Risk Drivers")
    
    if feature_importance is not None:
        top_5 = feature_importance.head(5)
        
        fig = go.Figure(go.Bar(
            x=top_5['Importance'],
            y=top_5['Feature'],
            orientation='h',
            marker=dict(
                color=['#1e3a8a', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd'],
                showscale=False
            ),
            text=top_5['Importance'],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Feature Importance (Model Decision Drivers)",
            xaxis_title="Importance Score",
            yaxis_title="",
            height=350,
            yaxis=dict(autorange="reversed"),
            showlegend=False
        )
        
        st.plotly_chart(fig, width="expand")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **1. To Bank (26.5%)**
            - Destination bank identifier
            - Bank 1004: High risk | Bank 800: Clean
            
            **2. From Bank (22.5%)**
            - Source bank identifier
            - Network effect analysis
            
            **3. Hour (19.1%)**
            - Time of day pattern
            - Peak risk: 11 AM - 3 PM
            """)
        
        with col2:
            st.markdown("""
            **4. Amount Received (15.5%)**
            - Transaction size
            - Structuring detection
            
            **5. Amount Paid (14.5%)**
            - Payment amount
            - $9K-$10K range critical
            """)

# ============================================================================
# TAB 2: TECHNICAL DEEP DIVE
# ============================================================================
with tab2:
    st.markdown("### Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **IBM Synthetic AML Dataset**
        - Total: 31,898,238 transactions
        - Laundering: 35,230 (0.11%)
        - Normal: 31,863,008 (99.89%)
        - Imbalance Ratio: 1:905
        - Period: Sept 1-28, 2022
        """)
    
    with col2:
        st.markdown("""
        **Feature Engineering**
        - Original features: 11
        - Engineered features: 21 new
        - Final features: 20 (validated)
        - Statistical tests: Chi-squared, ANOVA
        - All features: p < 0.001
        """)
    
    with col3:
        st.markdown("""
        **Training Strategy**
        - Train: 25.5M (80%)
        - Test: 6.4M (20%)
        - Split: Time-based
        - Scaling: StandardScaler
        - Training time: ~15 minutes
        """)
    
    st.markdown("---")
    
    # Model Comparison - Top 2 Models
    st.markdown("### Model Comparison: Top 2 Performers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Champion: Calibrated LightGBM")
        st.success("""
        **Configuration:**
        - Algorithm: LightGBM + CalibratedClassifierCV
        - Training: Full 25.5M samples
        - Threshold: 0.10 (optimized)
        - No class weighting (scale_pos_weight=1)
        
        **Performance:**
        - Precision: 72.75% (14.5x target)
        - Recall: 18.87%
        - F1 Score: 29.97% (3.0x target)
        - ROC-AUC: 97.81%
        - False Positive Rate: 0.01%
        
        **Why It Won:**
        - Probability calibration critical for reliability
        - Threshold optimization (0.10) maximizes F1 and precision
        - Full data training (no undersampling loss)
        - Sophisticated hyperparameters (1000 estimators, depth=8)
        - 27x reduction in false positives vs baseline
        """)
    
    with col2:
        st.markdown("#### Alternative: Cost-Sensitive Learning")
        st.info("""
        **Configuration:**
        - Algorithm: LightGBM with custom cost matrix
        - Training: Full 25.5M samples
        - Cost: FN=100x FP (regulatory focus)
        - Threshold: 0.70
        
        **Performance:**
        - Precision: 17.83%
        - Recall: 45.76%
        - F1 Score: 25.66%
        - ROC-AUC: 97.63%
        
        **Why It's 2nd:**
        - Higher recall (45.76% vs 18.87%)
        - Better for regulatory compliance focus
        - Lower precision (17.83% vs 72.75%)
        - 3x more alerts than champion
        - Alternative deployment option for high-recall needs
        """)
    
    st.markdown("---")
    
    # Why Other Models Failed
    st.markdown("### Why Other Models Failed")
    
    failure_tab1, failure_tab2, failure_tab3 = st.tabs(["Class Imbalance Failures", "Unsupervised Failure", "Undersampling Issues"])
    
    with failure_tab1:
        st.markdown('<div class="failed-label">FAILED: Class Weighting Alone</div>', unsafe_allow_html=True)
        st.error("""
        **Model 1 (XGBoost + Class Weight):**
        - Precision: 0.95% (too low)
        - Recall: 97.97% (too high)
        - Issue: 99% false positive rate - unusable
        - Reason: scale_pos_weight alone can't handle 1:905 imbalance
        
        **Model 3 (LightGBM + Class Weight):**
        - Precision: 0.98%
        - ROC-AUC: 85.42% (poor discrimination)
        - Issue: Model too aggressive, poor calibration
        
        **Model 7b (Cost-Sensitive Undersampled):**
        - Precision: 0.69%
        - Recall: 99.71% (flags everything)
        - Issue: Undersampling + cost matrix = too aggressive
        
        **Key Learning:** Class weighting requires calibration + threshold tuning
        """)
    
    with failure_tab2:
        st.markdown('<div class="failed-label">FAILED: Unsupervised Learning</div>', unsafe_allow_html=True)
        st.error("""
        **Isolation Forest**
        - Precision: 1.26% (58x worse than champion)
        - False Positives: ~390,000 (vs 752 for champion)
        - ROC-AUC: 87.60% (vs 97.81% for champion)
        
        **Root Cause:**
        - Trained only on normal transactions
        - No labeled laundering patterns to learn from
        - Laundering transactions too similar to normal
        - Unsupervised can't capture subtle patterns
        
        **Conclusion:** Supervised learning essential for AML detection
        """)
    
    with failure_tab3:
        st.markdown('<div class="failed-label">FAILED: Undersampling</div>', unsafe_allow_html=True)
        st.warning("""
        **Model 2b (XGBoost + 1:10 Undersampling):**
        - Precision: 7.20% | F1: 13.02%
        - Issue: Data loss from undersampling
        
        **Model 4 (LightGBM + 1:10 Undersampling):**
        - Precision: 7.12% | F1: 12.89%
        - Issue: Similar to Model 2b, no advantage
        
        **Model 6 (CatBoost + 1:10 Undersampling):**
        - Precision: 6.76% | F1: 12.31%
        - Issue: Worst of the undersampled models
        
        **Key Learning:** Undersampling reduces training data from 25.5M to 270K, 
        losing valuable patterns. Full data training (champion model) performs better.
        """)
    
    st.markdown("---")
    
    # Model Architecture
    st.markdown("### Model Architecture & Training Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Algorithm")
        st.markdown("""
        **Calibrated LightGBM**
        - Gradient boosting framework
        - Probability calibration (sigmoid)
        - 1,000 estimators
        - Max depth: 8
        - Learning rate: 0.05
        """)
    
    with col2:
        st.markdown("#### Training")
        if model_config:
            training_info = model_config['training_info']
            st.markdown(f"""
            **Data Split:**
            - Training: {training_info['training_samples']:,} samples
            - Testing: {training_info['test_samples']:,} samples
            - Features: {training_info['num_features']}
            - Time-based split (80/20)
            """)
    
    with col3:
        st.markdown("#### Performance")
        if model_config:
            st.markdown(f"""
            **Metrics:**
            - Precision: {metrics['precision']*100:.2f}%
            - Recall: {metrics['recall']*100:.2f}%
            - F1 Score: {metrics['f1_score']*100:.2f}%
            - ROC-AUC: {metrics['roc_auc']*100:.2f}%
            - FP Rate: 0.01%
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem 0;">
    <p><strong>Team: Delphin Kaduli, Tycho Janssen, Solomon Pinto</strong></p>
    <p><strong>Catholic University of America - MDA Capstone Project Fall'25</strong></p>
    <p>Anti-Money Laundering Detection using Machine Learning</p>
</div>
""", unsafe_allow_html=True)