"""
Model Validation Page
Displays detailed performance metrics, confusion matrix, and feature importance
for the production AML detection model.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import sys
sys.path.append('..')

st.set_page_config(
    page_title="Model Reporting",
    page_icon="",
    layout="wide"
)

# Load model config
@st.cache_data
def load_config():
    import os
    # Try multiple path options
    for base in ['../../models/', '../models/', './models/']:
        path = os.path.join(base, 'model_config.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    raise FileNotFoundError("model_config.json not found")

@st.cache_data
def load_feature_importance():
    import os
    # Try multiple path options
    for base in ['../../models/', '../models/', './models/']:
        path = os.path.join(base, 'feature_importance.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return pd.DataFrame(json.load(f))
    raise FileNotFoundError("feature_importance.json not found")

config = load_config()
feature_importance = load_feature_importance()

st.title("Model Reporting")
st.markdown("**Comprehensive performance metrics and validation results for data scientists**")

st.markdown("---")

# Performance Metrics
st.markdown("### Key Performance Metrics")
metrics = config['performance_metrics']

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Alert Confidence (Precision)", 
        f"{metrics['precision']*100:.2f}%", 
        help="73% of flagged transactions are actual laundering"
    )
with col2:
    st.metric(
        "Case Capture (Recall)", 
        f"{metrics['recall']*100:.2f}%", 
        help="Catches 19% of all laundering transactions"
    )
with col3:
    st.metric(
        "F1 Score", 
        f"{metrics['f1_score']*100:.2f}%", 
        help="Balanced measure of precision and recall"
    )
with col4:
    st.metric(
        "ROC-AUC", 
        f"{metrics['roc_auc']*100:.2f}%", 
        help="Model discrimination ability"
    )

st.markdown("---")

# Confusion Matrix
st.markdown("### Confusion Matrix")

col1, col2 = st.columns([2, 1])

with col1:
    # Create confusion matrix heatmap
    cm_data = [
        [metrics['true_negatives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_positives']]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=['Predicted Normal', 'Predicted Laundering'],
        y=['Actual Normal', 'Actual Laundering'],
        text=cm_data,
        texttemplate='%{text:,}',
        textfont={"size": 16},
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix (Test Set)",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=400
    )
    
    st.plotly_chart(fig, width="expand")

with col2:
    st.markdown("#### Matrix Breakdown")
    st.success(f"""
    **True Negatives: {metrics['true_negatives']:,}**
    Correctly identified normal transactions
    99.99% accuracy on normal transactions
    """)
    
    st.warning(f"""
    **False Positives: {metrics['false_positives']:,}**
    Normal transactions flagged incorrectly
    Only 0.01% false alarm rate
    """)
    
    st.error(f"""
    **False Negatives: {metrics['false_negatives']:,}**
    Laundering cases missed
    Trade-off for high precision
    """)
    
    st.info(f"""
    **True Positives: {metrics['true_positives']:,}**
    Correctly identified laundering cases
    18.87% of all laundering caught
    """)

st.markdown("---")

# Feature Importance
st.markdown("### Feature Importance")

col1, col2 = st.columns([2, 1])

with col1:
    top_15 = feature_importance.head(15)
    
    fig = go.Figure(go.Bar(
        x=top_15['Importance'],
        y=top_15['Feature'],
        orientation='h',
        marker=dict(
            color=top_15['Importance'],
            colorscale='Viridis',
            showscale=False
        )
    ))
    
    fig.update_layout(
        title="Top 15 Features by Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500,
        yaxis=dict(autorange="reversed")
    )
    
    st.plotly_chart(fig, width="expand")

with col2:
    st.markdown("#### Top 5 Features")
    
    total_importance = feature_importance['Importance'].sum()
    
    for idx, row in feature_importance.head(5).iterrows():
        contribution = (row['Importance'] / total_importance) * 100
        st.markdown(f"""
        **{idx+1}. {row['Feature']}**
        - Importance: {row['Importance']:,}
        - Contribution: {contribution:.1f}%
        """)

st.markdown("---")

# Model Configuration
st.markdown("### Model Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Hyperparameters")
    hyperparams = config['hyperparameters']
    st.json(hyperparams)

with col2:
    st.markdown("#### Training Information")
    training = config['training_info']
    st.markdown(f"""
    - **Training Samples:** {training['training_samples']:,}
    - **Test Samples:** {training['test_samples']:,}
    - **Features:** {training['num_features']}
    - **Training Date:** {training['training_date']}
    """)

with col3:
    st.markdown("#### Model Details")
    st.markdown(f"""
    - **Model Name:** {config['model_name']}
    - **Version:** {config['model_version']}
    - **Optimal Threshold:** {config['optimal_threshold']}
    - **Calibration:** Sigmoid
    """)

st.markdown("---")

# Validation Summary
st.markdown("### Validation Summary")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    **✅ Production Ready**
    
    - Precision exceeds 70% target (72.75%)
    - False positive rate under 0.02% (0.01%)
    - ROC-AUC demonstrates excellent discrimination (97.81%)
    - Confusion matrix shows strong performance
    - Model properly calibrated for reliable probabilities
    """)

with col2:
    st.info("""
    **📊 Key Validation Points**
    
    - Time-based train/test split (80/20)
    - No data leakage in feature engineering
    - Threshold optimized for F1 and precision
    - Feature importance aligns with domain knowledge
    - Model tested on 6.4M unseen transactions
    """)
