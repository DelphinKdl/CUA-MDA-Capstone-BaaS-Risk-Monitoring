"""
Risk Patterns Analysis
Deep dive into feature importance, risk factors, and pattern analysis
for AML detection model.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json

st.set_page_config(
    page_title="Risk Intelligence",
    page_icon="",
    layout="wide"
)

# Load feature importance
@st.cache_data
def load_feature_data():
    import os
    # Try multiple path options
    for base in ['../../models/', '../models/', './models/']:
        importance_path = os.path.join(base, 'feature_importance.json')
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                importance = pd.DataFrame(json.load(f))
            with open(os.path.join(base, 'top_15_features.json'), 'r') as f:
                top_features = json.load(f)
            return importance, top_features
    raise FileNotFoundError("Feature files not found")

feature_importance, top_15_features = load_feature_data()

st.title(" Risk Intelligence")
st.markdown("**Feature importance and risk pattern analysis for AML detection**")

st.markdown("---")

# Feature Importance Overview
st.markdown("### Feature Importance Distribution")

col1, col2 = st.columns([3, 1])

with col1:
    # Create treemap of feature importance
    fig = px.treemap(
        feature_importance.head(15),
        path=['Feature'],
        values='Importance',
        title='Feature Importance Treemap (Top 15)',
        color='Importance',
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="expand")

with col2:
    st.markdown("#### Summary Statistics")
    total_importance = feature_importance['Importance'].sum()
    top_5_importance = feature_importance.head(5)['Importance'].sum()
    top_10_importance = feature_importance.head(10)['Importance'].sum()
    
    st.metric("Total Features", len(feature_importance))
    st.metric("Top 5 Contribution", f"{(top_5_importance/total_importance)*100:.1f}%")
    st.metric("Top 10 Contribution", f"{(top_10_importance/total_importance)*100:.1f}%")

st.markdown("---")

# Detailed Feature Rankings
st.markdown("### Feature Rankings")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["All Features", "Top 15 Features", "Feature Categories"])

with tab1:
    # Display full feature importance table
    display_df = feature_importance.copy()
    display_df['Contribution %'] = (display_df['Importance'] / total_importance * 100).round(2)
    display_df['Rank'] = range(1, len(display_df) + 1)
    
    st.dataframe(
        display_df[['Rank', 'Feature', 'Importance', 'Contribution %']],
        use_container_width=True,
        height=600
    )

with tab2:
    st.markdown("#### Top 15 Most Important Features for Production Model")
    
    top_15_df = feature_importance.head(15).copy()
    top_15_df['Contribution %'] = (top_15_df['Importance'] / total_importance * 100).round(2)
    top_15_df['Cumulative %'] = top_15_df['Contribution %'].cumsum().round(2)
    
    for idx, row in top_15_df.iterrows():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**{idx+1}. {row['Feature']}**")
        with col2:
            st.markdown(f"Importance: {row['Importance']:,}")
        with col3:
            st.markdown(f"Contribution: {row['Contribution %']:.1f}%")

with tab3:
    st.markdown("#### Features by Category")
    
    # Categorize features
    categories = {
        'Bank Identifiers': ['To Bank', 'From Bank', 'is_bank_800', 'is_bank_1004'],
        'Transaction Amounts': ['Amount Received', 'Amount Paid', 'amount_zscore'],
        'Temporal Features': ['hour', 'day_of_week', 'is_weekend', 'is_night'],
        'Payment Methods': ['is_ach', 'ach_weekend'],
        'Currency Flags': ['is_usd', 'is_euro', 'is_uk_pound'],
        'Risk Indicators': ['risk_score_v2', 'in_structuring_range', 'is_just_below_threshold', 'uk_pound_structuring']
    }
    
    for category, features in categories.items():
        with st.expander(f"{category} ({len(features)} features)"):
            category_features = feature_importance[feature_importance['Feature'].isin(features)]
            if not category_features.empty:
                category_importance = category_features['Importance'].sum()
                category_pct = (category_importance / total_importance) * 100
                
                st.markdown(f"**Total Category Importance:** {category_importance:,} ({category_pct:.1f}%)")
                st.markdown("---")
                
                for _, row in category_features.iterrows():
                    pct = (row['Importance'] / total_importance) * 100
                    st.markdown(f"- **{row['Feature']}**: {row['Importance']:,} ({pct:.1f}%)")

st.markdown("---")

# Risk Factor Analysis
st.markdown("### Key Risk Factors")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Highest Risk Indicators")
    st.error("""
    **1. ACH Payment Format (49.7x baseline risk)**
    - Importance: 744
    - Most significant payment method indicator
    - 84% of all transactions use ACH
    - Enables structuring and smurfing patterns
    
    **2. Weekend Transactions (3.0x baseline risk)**
    - Importance: 344
    - Reduced oversight on weekends
    - Combined with ACH creates 21.6x risk
    
    **3. Bank 1004 Pattern**
    - Importance: 628
    - Consistently associated with laundering
    - Possible money mule network
    
    **4. UK Pound Structuring (8.3x baseline risk)**
    - Importance: 39
    - Amounts in $9K-$10K range
    - Avoiding $10K CTR threshold
    """)

with col2:
    st.markdown("#### Protective Factors")
    st.success("""
    **1. Bank 800 Transactions**
    - Importance: 1,092
    - Completely clean record
    - Zero laundering cases detected
    - Trusted institution indicator
    
    **2. USD Currency**
    - Importance: 2,493
    - Lower risk than exotic currencies
    - Standard business transactions
    
    **3. Euro Currency**
    - Importance: 2,346
    - Legitimate international trade
    - Lower risk profile
    
    **4. Night Transactions (0.53x baseline)**
    - Importance: 740
    - Actually safer than expected
    - Automated legitimate payments
    """)

st.markdown("---")

# Feature Interactions
st.markdown("### Feature Interactions")

st.info("""
**High-Risk Combinations:**

1. **ACH + Weekend** (21.6x risk)
   - Reduced monitoring on weekends
   - ACH enables rapid movement
   - Combined effect multiplies risk

2. **UK Pound + Structuring Range** (8.3x risk)
   - Currency-specific threshold avoidance
   - Amounts just below $10K reporting limit
   - Deliberate pattern to avoid detection

3. **Bank 1004 + High Amount** 
   - Compromised institution
   - Large transaction volumes
   - Network effect amplification

4. **Weekend + Night + ACH**
   - Minimal oversight window
   - Automated processing
   - Delayed detection opportunity
""")

st.markdown("---")

# Model Insights
st.markdown("### Model Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Bank-Level Patterns")
    st.info("""
    - **To Bank** and **From Bank** are top 2 features (49% combined importance)
    - Specific banks show consistent patterns
    - Bank-level risk scoring is critical
    - Network analysis opportunities
    """)

with col2:
    st.markdown("#### Temporal Patterns")
    st.info("""
    - **Hour** is 3rd most important (19.1%)
    - Peak risk: 11 AM - 3 PM
    - Weekend effect significant
    - Night transactions surprisingly safe
    """)

with col3:
    st.markdown("#### Amount-Based Detection")
    st.info("""
    - **Amount Received/Paid** in top 5 (30% combined)
    - Z-score normalization effective
    - Structuring detection works
    - Threshold avoidance patterns clear
    """)

st.markdown("---")

# Pattern Detection Guide
st.markdown("### Pattern Detection Guide for Analysts")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Red Flags to Watch")
    st.warning("""
    **Immediate Investigation Required:**
    
     Bank 1004 + Any amount > $5K
     ACH + Weekend + Structuring range
     UK Pound + $9K-$10K amounts
     Multiple transactions just below $10K
     Weekend + Night combinations
    
    **Escalation Criteria:**
    - Risk score > 50%
    - Multiple red flags present
    - Pattern repetition detected
    - Known high-risk banks involved
    """)

with col2:
    st.markdown("#### Safe Patterns")
    st.success("""
    **Low Priority for Review:**
    
     Bank 800 transactions (any amount)
     USD currency + Business hours
     Night transactions (automated)
     Weekday + Standard amounts
     No structuring indicators
    
    **Confidence Indicators:**
    - Risk score < 5%
    - Trusted bank involved
    - Standard business patterns
    - No threshold avoidance
    """)
