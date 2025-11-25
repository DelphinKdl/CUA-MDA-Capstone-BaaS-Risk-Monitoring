"""
Data Insights & EDA Visualizations
Exploratory Data Analysis findings from the training dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Data Insights",
    page_icon="",
    layout="wide"
)

st.title("Data Insights")
st.markdown("**Exploratory Data Analysis from 31.9M transactions (IBM Synthetic AML Dataset)**")

st.markdown("---")

# Dataset Overview
st.markdown("### Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", "31.9M")
with col2:
    st.metric("Laundering Cases", "35,230")
with col3:
    st.metric("Normal Transactions", "31.9M")
with col4:
    st.metric("Imbalance Ratio", "1:905")

st.markdown("---")

# Class Distribution
st.markdown("### Class Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    # Class distribution pie chart
    labels = ['Normal Transactions', 'Money Laundering']
    values = [31863008, 35230]
    colors = ['#3b82f6', '#dc2626']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title="Severe Class Imbalance (Only 0.11% Laundering)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### The Challenge")
    st.error("""
**Extreme Imbalance:**
- Only 0.11% are laundering cases
- 905 normal transactions for every 1 laundering case
- Standard ML models fail on this data

**Our Approach:**
- Calibrated probabilities for reliable risk scores
- Optimized decision threshold (10%)
- Trained on full dataset (no undersampling)
    """)

st.markdown("---")

# Payment Format Distribution
st.markdown("### Payment Format Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    # Payment format bar chart
    payment_formats = ['ACH', 'Wire', 'Check', 'Cash', 'Bitcoin']
    normal_pct = [84.2, 8.5, 4.3, 2.1, 0.9]
    laundering_pct = [95.8, 2.1, 1.2, 0.7, 0.2]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Normal',
        x=payment_formats,
        y=normal_pct,
        marker_color='#3b82f6',
        text=[f'{v}%' for v in normal_pct],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Laundering',
        x=payment_formats,
        y=laundering_pct,
        marker_color='#dc2626',
        text=[f'{v}%' for v in laundering_pct],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Payment Format by Transaction Type",
        xaxis_title="Payment Format",
        yaxis_title="Percentage (%)",
        barmode='group',
        height=400,
        yaxis_range=[0, 110]
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Key Finding")
    st.warning("""
**ACH Payment Dominance:**
- 96% of laundering uses ACH payments
- vs 84% for normal transactions
- Significantly elevated risk

**Why ACH?**
- Fast processing
- Easy automation
- Enables structuring patterns
- Lower scrutiny than wire transfers
    """)

st.markdown("---")

# Temporal Patterns
st.markdown("### Temporal Patterns")

col1, col2 = st.columns([2, 1])

with col1:
    # Day of week distribution
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    normal_daily = [15.2, 15.5, 15.8, 15.3, 15.1, 11.8, 11.3]
    laundering_daily = [14.8, 14.5, 14.2, 14.9, 15.2, 13.5, 12.9]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Normal',
        x=days,
        y=normal_daily,
        marker_color='#3b82f6',
        text=[f'{v}%' for v in normal_daily],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Laundering',
        x=days,
        y=laundering_daily,
        marker_color='#dc2626',
        text=[f'{v}%' for v in laundering_daily],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Transaction Distribution by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Percentage (%)",
        barmode='group',
        height=400,
        yaxis_range=[0, 20]
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Weekend Pattern")
    st.info("""
**Weekend Activity:**
- Weekend laundering: 26.4%
- Weekend normal: 23.1%
- Elevated weekend risk

**Insight:**
- Reduced oversight on weekends
- Combined with ACH, risk increases significantly
- Key temporal indicator for detection
    """)

st.markdown("---")

# Amount Distribution
st.markdown("### Transaction Amount Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    # Amount distribution histogram
    amount_ranges = ['$0-1K', '$1K-5K', '$5K-9K', '$9K-10K', '$10K-50K', '$50K+']
    normal_amounts = [35.2, 28.5, 15.3, 2.1, 12.8, 6.1]
    laundering_amounts = [18.5, 22.3, 28.5, 18.2, 8.5, 4.0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Normal',
        x=amount_ranges,
        y=normal_amounts,
        marker_color='#3b82f6',
        text=[f'{v}%' for v in normal_amounts],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Laundering',
        x=amount_ranges,
        y=laundering_amounts,
        marker_color='#dc2626',
        text=[f'{v}%' for v in laundering_amounts],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Transaction Amount Distribution",
        xaxis_title="Amount Range",
        yaxis_title="Percentage (%)",
        barmode='group',
        height=400,
        yaxis_range=[0, 40]
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Structuring Detection")
    st.error("""
**Range (Structuring):**
- 18% of laundering transactions
- vs 2% of normal transactions
- Significantly elevated risk

**CTR Threshold Avoidance:**
- $10,000 = Currency Transaction Report threshold
- Criminals structure amounts just below this limit
- Clear evasion pattern detected

**Model Impact:**
- Structuring range is a key predictive feature
- Combined with currency patterns for enhanced detection
    """)

st.markdown("---")

# Summary
st.markdown("### Key Takeaways")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
**Payment Patterns**
- ACH payments dominate laundering activity
- Bitcoin is NOT the primary risk
- Payment method is highly predictive
    """)

with col2:
    st.info("""
**Temporal Signals**
- Weekend activity shows elevated risk
- Combined patterns amplify detection
- Time-based features are valuable
    """)

with col3:
    st.info("""
**Amount Structuring**
- Nine to ten thousand range is critical indicator
- Threshold avoidance behavior detected
- Amount-based features essential
    """)