"""
Investigator Workbench
Real-time transaction risk scoring tool for compliance analysts.
Score individual transactions and identify risk factors.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(
    page_title="AML Prediction",
    page_icon="",
    layout="wide"
)

# Load model artifacts
@st.cache_resource
def load_model():
    import os
    for base in ['../../models/', '../models/', './models/']:
        model_path = os.path.join(base, 'calibrated_lightgbm_model.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            scaler = joblib.load(os.path.join(base, 'scaler.pkl'))
            with open(os.path.join(base, 'feature_names.json'), 'r') as f:
                features = json.load(f)
            with open(os.path.join(base, 'model_config.json'), 'r') as f:
                config = json.load(f)
            return model, scaler, features, config
    raise FileNotFoundError("Model files not found")

model, scaler, feature_names, config = load_model()

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

st.title(" Transaction Monitoring")
st.markdown("**Real-time transaction risk scoring and case investigation for compliance team**")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Transaction Timing**")
    from datetime import datetime
    transaction_date = st.date_input("Transaction Date", value=datetime.now())
    transaction_time = st.time_input("Transaction Time", value=datetime.now().time())
    
with col2:
    st.markdown("**Transaction Amount**")
    amount = st.number_input("Amount ($)", min_value=0.0, value=5000.0, step=100.0, 
                            help="Transaction amount in dollars")
    payment_currency = st.selectbox("Payment Currency", 
                                    ["US Dollar", "Euro", "UK Pound", "Yen", "Yuan", "Bitcoin", 
                                     "Australian Dollar", "Brazil Real", "Canadian Dollar", 
                                     "Mexican Peso", "Ruble", "Rupee", "Saudi Riyal", 
                                     "Shekel", "Swiss Franc"],
                                    index=0)
    
with col3:
    st.markdown("**Payment Method**")
    payment_format = st.selectbox("Payment Format", 
                                 ["ACH", "Wire", "Cheque", "Cash", "Bitcoin", "Credit Card", "Reinvestment"],
                                 index=1)
    receiving_currency = st.selectbox("Receiving Currency",
                                     ["US Dollar", "Euro", "UK Pound", "Yen", "Yuan", "Bitcoin", 
                                      "Australian Dollar", "Brazil Real", "Canadian Dollar", 
                                      "Mexican Peso", "Ruble", "Rupee", "Saudi Riyal", 
                                      "Shekel", "Swiss Franc"],
                                     index=0)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Sender Information**")
    sender_bank = st.number_input("Sender Bank ID", min_value=0, value=12345, step=1,
                                 help="Bank ID of the sender")
    sender_account = st.text_input("Sender Account", value="ACC-123456",
                                   help="Sender account number (for reference only)")

with col2:
    st.markdown("**Receiver Information**")
    receiver_bank = st.number_input("Receiver Bank ID", min_value=0, value=67890, step=1,
                                    help="Bank ID of the receiver")
    receiver_account = st.text_input("Receiver Account", value="ACC-789012",
                                     help="Receiver account number (for reference only)")

# Score Transaction Button
if st.button(" Score Transaction", type="primary", use_container_width=True):
    st.markdown("---")
    st.markdown("###  Risk Assessment Results")
    
    # Extract temporal features from datetime
    hour = transaction_time.hour
    day_of_week = transaction_date.weekday()  # 0=Monday, 6=Sunday
    is_weekend = 1 if day_of_week >= 5 else 0
    is_night = 1 if hour >= 22 or hour < 6 else 0
    
    # Payment format features
    is_ach = 1 if payment_format == "ACH" else 0
    
    # Currency features
    is_usd = 1 if payment_currency == "US Dollar" else 0
    is_euro = 1 if payment_currency == "Euro" else 0
    is_uk_pound = 1 if payment_currency == "UK Pound" else 0
    
    # Bank-specific features (based on account prefix)
    sender_account_str = str(sender_account)
    receiver_account_str = str(receiver_account)
    
    # Check if account/bank starts with 800 or 1004
    is_bank_800 = 1 if (sender_account_str.startswith('800') or 
                        receiver_account_str.startswith('800') or
                        str(sender_bank).startswith('800') or 
                        str(receiver_bank).startswith('800')) else 0
    
    is_bank_1004 = 1 if (sender_account_str.startswith('1004') or 
                         receiver_account_str.startswith('1004') or
                         str(sender_bank).startswith('1004') or 
                         str(receiver_bank).startswith('1004')) else 0
    
    # Structuring detection - CORRECTED
    # is_just_below_threshold: 9K-10K for major currencies
    is_just_below_threshold = 0
    if payment_currency in ['US Dollar', 'Euro', 'UK Pound', 'Canadian Dollar', 'Australian Dollar']:
        if 9000 <= amount < 10000:
            is_just_below_threshold = 1
    
    # in_structuring_range: 3K-9K (from training)
    in_structuring_range = 1 if 3000 <= amount <= 9000 else 0
    
    # Combined risk patterns
    ach_weekend = is_ach * is_weekend
    uk_pound_structuring = is_uk_pound * is_just_below_threshold
    
    # Statistical features - CORRECTED with actual dataset statistics
    mean_amount = 5392240  # Actual mean from dataset
    std_amount = 1298679000  # Actual std from dataset
    amount_zscore = (amount - mean_amount) / std_amount
    
    # Risk score v2: composite risk indicator
    risk_score_v2 = (
        is_ach * 3.0 +
        is_weekend * 1.5 +
        in_structuring_range * 4.0 +
        is_bank_1004 * 5.0 +
        is_uk_pound * 2.0
    )
    
    # Prepare input data using exact feature names from training
    input_dict = {
        'To Bank': receiver_bank,
        'From Bank': sender_bank,
        'Amount Received': amount,
        'Amount Paid': amount,
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_night': is_night,
        'is_ach': is_ach,
        'is_usd': is_usd,
        'is_euro': is_euro,
        'is_uk_pound': is_uk_pound,
        'is_bank_800': is_bank_800,
        'is_bank_1004': is_bank_1004,
        'in_structuring_range': in_structuring_range,
        'is_just_below_threshold': is_just_below_threshold,
        'ach_weekend': ach_weekend,
        'uk_pound_structuring': uk_pound_structuring,
        'amount_zscore': amount_zscore,
        'risk_score_v2': risk_score_v2
    }
    
    #  features 
    input_data = pd.DataFrame([input_dict], columns=feature_names)
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Get prediction
    risk_probability = model.predict_proba(input_scaled)[0][1]
    threshold = config['optimal_threshold']
    prediction = 1 if risk_probability >= threshold else 0
    
    #
    prediction_record = {
        'Timestamp': f"{transaction_date} {transaction_time}",
        'Sender Bank': sender_bank,
        'Receiver Bank': receiver_bank,
        'Amount': f"${amount:,.2f}",
        'Currency': payment_currency,
        'Payment Format': payment_format,
        'Risk Score': f"{risk_probability*100:.2f}%",
        'Status': 'HIGH RISK' if prediction == 1 else 'LOW RISK'
    }
    st.session_state.prediction_history.append(prediction_record)
 
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Risk Probability",
            f"{risk_probability*100:.2f}%",
            help="Model confidence that this is laundering"
        )
    
    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold*100:.0f}%",
            help="Optimized threshold for production"
        )
    
    with col3:
        if prediction == 1:
            st.error("**HIGH RISK**")
            st.markdown("Flag for investigation")
        else:
            st.success("**LOW RISK**")
            st.markdown("Normal transaction")
    
    st.markdown("---")
    
    # Detailed Assessment
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Assessment")
        
        if prediction == 1:
            st.error(f"""
**HIGH RISK ALERT**

Risk Score: **{risk_probability*100:.2f}%** (Above 10% threshold)

**Recommended Actions:**
- Flag for immediate investigation
- Review customer transaction history
- Consider SAR filing if patterns confirmed
""")
        else:
            st.success(f"""
**LOW RISK TRANSACTION**

Risk Score: **{risk_probability*100:.2f}%** (Below 10% threshold)

**Recommended Actions:**
- No immediate action required
- Continue routine monitoring
""")
    
    with col2:
        st.markdown("### Key Indicators")
        
        if prediction == 1:
            risk_factors = []
            
            if is_ach:
                risk_factors.append("ACH payment format")
            if is_weekend:
                risk_factors.append("Weekend transaction")
            if in_structuring_range:
                risk_factors.append("Structuring pattern")
            if is_bank_1004:
                risk_factors.append("High-risk institution")
            if is_uk_pound and in_structuring_range:
                risk_factors.append("Currency risk pattern")
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(factor)
            else:
                st.info("Multiple minor risk signals detected")
        else:
            st.success("Transaction cleared")
            st.info("No suspicious patterns detected")