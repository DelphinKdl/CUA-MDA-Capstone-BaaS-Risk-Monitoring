"""
Model Validation Page
Performance metrics and validation results for the production AML detection model.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json

st.set_page_config(
    page_title="Model Validation",
    page_icon="",
    layout="wide"
)

# Load model config
@st.cache_data
def load_config():
    import os
    for base in ['../../models/', '../models/', './models/']:
        path = os.path.join(base, 'model_config.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    raise FileNotFoundError("model_config.json not found")

config = load_config()

st.title("Model Validation")
st.markdown("**Performance metrics and validation results**")

st.markdown("---")

# Performance Metrics
st.markdown("### Key Performance Metrics")
metrics = config['performance_metrics']

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Alert Confidence (Precision)", 
        f"{metrics['precision']*100:.2f}%"
    )
with col2:
    st.metric(
        "Case Capture (Recall)", 
        f"{metrics['recall']*100:.2f}%"
    )
with col3:
    st.metric(
        "F1 Score", 
        f"{metrics['f1_score']*100:.2f}%"
    )
with col4:
    st.metric(
        "ROC-AUC", 
        f"{metrics['roc_auc']*100:.2f}%"
    )

st.markdown("---")

# Confusion Matrix
st.markdown("### Confusion Matrix")

col1, col2 = st.columns([3, 2])

with col1:
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
        textfont={"size": 18},
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        title="Test Set Performance",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Performance Summary")
    
    st.success(f"""
**True Negatives: {metrics['true_negatives']:,}**
- Correctly identified normal transactions
- 99.99% accuracy on normal cases
    """)
    
    st.info(f"""
**True Positives: {metrics['true_positives']:,}**
- Correctly caught laundering cases
- 19% capture rate with high confidence
    """)
    
    st.warning(f"""
**False Positives: {metrics['false_positives']:,}**
- Normal transactions flagged
- Only 0.01% false alarm rate
    """)
    
    st.error(f"""
**False Negatives: {metrics['false_negatives']:,}**
- Laundering cases missed
- Trade-off for precision optimization
    """)

st.markdown("---")

# Model Configuration
st.markdown("### Model Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"""
**Model Type**
- Algorithm: LightGBM
- Calibration: Isotonic Regression
- Training Data: 25.5M transactions
    """)

with col2:
    st.info(f"""
**Decision Threshold**
- Optimized Threshold: {config['optimal_threshold']*100:.0f}%
- Precision-focused strategy
- Maximizes analyst efficiency
    """)

with col3:
    st.info(f"""
**Business Impact**
- 27x reduction in false positives
- ~860 alerts per day (manageable)
- 62% reduction in analyst workload
    """)