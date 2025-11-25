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


st.sidebar.caption("CUA MDA Capstone Project Fall'25")
st.sidebar.caption("Team: Delphin Kaduli, Tycho Janssen, Solomon Pinto")

# Main Page Header
st.markdown('<div class="main-header">Anti Money Laundering Detection System</div>', unsafe_allow_html=True)
st.markdown("**Production-Ready AML Detection for Banking Compliance**")
st.markdown("---")

st.markdown("### Key Performance Indicators")
    
if model_config:
    metrics = model_config['performance_metrics']
    col1, col2, col3= st.columns(3)
    
    with col1:
        st.metric(
            label="Alert Confidence (Precision)",
            value=f"{metrics['precision']*100:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Case Capture Rate (Recall)",
                value=f"{metrics['recall']*100:.2f}%"
        )
    
    with col3:
        st.metric(
            label="ROC-AUC",
            value=f"{metrics['roc_auc']*100:.2f}%"
        )
    

# Model - Business Focus
st.markdown("### The Best Model")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    **LightGBM + Probability Calibration**
    - Decision Threshold: 0.10 (optimized for precision)
    - Training: Full 25.5M transactions
    **Key Advantage:** 
Calibrated probabilities, when model says 10% risk, it means 10%
    
  
    """)

with col2:
    st.info("""
      **Business Impact:**
    - Reduces analyst workload by 62% vs alternative models
    - 27x reduction in false positives
    - 73% confidence in every alert
    - Manageable 860 alerts per day
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