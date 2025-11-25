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
    page_title="Case Management",
    page_icon="",
    layout="wide"
)

# Load model artifacts
@st.cache_resource
def load_model():
    import os
    # Try multiple path options
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

# Initialize session state and load from Excel if exists
if 'prediction_history' not in st.session_state:
    import os
    excel_file = 'prediction_history.xlsx'
    if os.path.exists(excel_file):
        try:
            existing_df = pd.read_excel(excel_file)
            st.session_state.prediction_history = existing_df.to_dict('records')
        except:
            st.session_state.prediction_history = []
    else:
        st.session_state.prediction_history = []

st.title("Transaction monitoring")
st.markdown("**Real-time transaction risk scoring and case investigation for compliance analysts**")

st.markdown("---")

# Input Form - User-Friendly Transaction Entry
st.markdown("### Enter Transaction Details for Prediction")
st.info("Enter transaction information as it appears in your system. All risk features will be calculated automatically.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Transaction Timing**")
    from datetime import datetime
    transaction_date = st.date_input("Transaction Date", value=datetime.now())
    transaction_time = st.time_input("Transaction Time", value=datetime.now().time())
    
with col2:
    st.markdown("**Transaction Amount**")
    amount = st.number_input("Amount ($)", min_value=0.0, value=9450.0, step=100.0, 
                            help="Transaction amount in dollars")
    payment_currency = st.selectbox("Payment Currency", 
                                    ["US Dollar", "Euro", "UK Pound", "Yen", "Yuan", "Bitcoin", 
                                     "Australian Dollar", "Brazil Real", "Canadian Dollar", 
                                     "Mexican Peso", "Ruble", "Rupee", "Saudi Riyal", 
                                     "Shekel", "Swiss Franc"],
                                    index=2)
    
with col3:
    st.markdown("**Payment Method**")
    payment_format = st.selectbox("Payment Format", 
                                 ["ACH", "Wire", "Cheque", "Cash", "Bitcoin", "Credit Card", "Reinvestment"],
                                 index=0)
    receiving_currency = st.selectbox("Receiving Currency",
                                     ["US Dollar", "Euro", "UK Pound", "Yen", "Yuan", "Bitcoin", 
                                      "Australian Dollar", "Brazil Real", "Canadian Dollar", 
                                      "Mexican Peso", "Ruble", "Rupee", "Saudi Riyal", 
                                      "Shekel", "Swiss Franc"],
                                     index=2)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Sender Information**")
    sender_bank = st.number_input("Sender Bank ID", min_value=0, value=800072000, step=1,
                                 help="Bank ID of the sender")
    sender_account = st.text_input("Sender Account", value="ACC-123456",
                                   help="Sender account number (for reference only)")

with col2:
    st.markdown("**Receiver Information**")
    receiver_bank = st.number_input("Receiver Bank ID", min_value=0, value=100428660, step=1,
                                    help="Bank ID of the receiver")
    receiver_account = st.text_input("Receiver Account", value="ACC-789012",
                                     help="Receiver account number (for reference only)")

# Score Transaction Button
if st.button("Score Transaction", use_container_width=True):
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
    
    # Bank-specific features
    is_bank_800 = 1 if receiver_bank == 800 or sender_bank == 800 else 0
    is_bank_1004 = 1 if receiver_bank == 1004 or sender_bank == 1004 else 0
    
    # Structuring detection (amounts between $9K-$10K)
    in_structuring_range = 1 if 9000 <= amount <= 10000 else 0
    is_just_below_threshold = 1 if 9500 <= amount < 10000 else 0
    
    # Combined risk patterns
    ach_weekend = is_ach * is_weekend
    uk_pound_structuring = is_uk_pound * in_structuring_range
    
    # Statistical features (simplified calculations)
    # Z-score: how many standard deviations from mean transaction amount
    mean_amount = 5000  # Average transaction amount
    std_amount = 3000   # Standard deviation
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
        'Amount Paid': amount,  # Assuming same amount
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
    
    # Create DataFrame with features in the exact order from training
    input_data = pd.DataFrame([input_dict], columns=feature_names)
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Get prediction
    risk_probability = model.predict_proba(input_scaled)[0][1]
    threshold = config['optimal_threshold']
    prediction = 1 if risk_probability >= threshold else 0
    
    # Save to prediction history
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
    
    # Auto-save to Excel file
    import os
    excel_file = 'prediction_history.xlsx'
    history_df_save = pd.DataFrame(st.session_state.prediction_history)
    
    try:
        # Check if file exists and append, otherwise create new
        if os.path.exists(excel_file):
            # Read existing data
            existing_df = pd.read_excel(excel_file)
            # Append new record
            updated_df = pd.concat([existing_df, pd.DataFrame([prediction_record])], ignore_index=True)
            updated_df.to_excel(excel_file, index=False)
        else:
            # Create new file
            history_df_save.to_excel(excel_file, index=False)
        
        st.success(f" Prediction automatically saved to {excel_file}")
    except Exception as e:
        st.warning(f"Could not auto-save to Excel: {e}")
    
    # Display Results
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
    
    # Show automatically detected risk features
    st.markdown("###  Detected Risk Features")
    st.caption("These features were automatically extracted from your transaction:")
    
    risk_features = []
    if is_ach:
        risk_features.append("ACH Payment (49.7x baseline risk)")
    if is_weekend:
        risk_features.append("Weekend Transaction (3.0x baseline risk)")
    if in_structuring_range:
        risk_features.append(" Structuring Range 9K-$10K (8.3x baseline risk)")
    if is_bank_1004:
        risk_features.append(" High-Risk Bank 1004 (95.2% laundering rate)")
    if is_uk_pound:
        risk_features.append(" UK Pound Currency")
    if ach_weekend:
        risk_features.append(" ACH + Weekend Combined (21.6x baseline risk)")
    if uk_pound_structuring:
        risk_features.append(" UK Pound + Structuring (8.3x baseline risk)")
    if is_night:
        risk_features.append(" Night Transaction (10PM-6AM)")
    
    if risk_features:
        for feature in risk_features:
            st.warning(feature)
    else:
        st.success(" No high-risk features detected")
    
    st.markdown("---")
    
    # Detailed Assessment
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Analysis")
        
        if prediction == 1:
            st.error(f"""
            **ALERT: Potential Money Laundering**
            
            - Risk Score: {risk_probability*100:.2f}%
            - Exceeds threshold: {threshold*100:.0f}%
            - Confidence: {(risk_probability/threshold)*100:.1f}% above threshold
            
            **Recommended Action:**
            - Immediate investigation required
            - Review transaction history
            - Check for related patterns
            - Consider SAR filing if confirmed
            """)
        else:
            st.success(f"""
            **NORMAL: Low Risk Transaction**
            
            - Risk Score: {risk_probability*100:.2f}%
            - Below threshold: {threshold*100:.0f}%
            - Safety margin: {((threshold-risk_probability)/threshold)*100:.1f}%
            
            **Recommended Action:**
            - No immediate action required
            - Continue routine monitoring
            - Transaction appears legitimate
            """)
    
    with col2:
        st.markdown("### Risk Factors Identified")
        
        risk_factors = []
        
        if is_ach:
            risk_factors.append(" ACH Payment (49x baseline risk)")
        if is_weekend:
            risk_factors.append(" Weekend Transaction (3x baseline risk)")
        if ach_weekend:
            risk_factors.append(" ACH + Weekend (21.6x combined risk)")
        if in_structuring_range:
            risk_factors.append(" Structuring Range $9K-$10K")
        if is_uk_pound and in_structuring_range:
            risk_factors.append(" UK Pound Structuring (8.3x risk)")
        if is_bank_1004:
            risk_factors.append(" Bank 1004 Pattern (High Risk)")
        if is_just_below_threshold:
            risk_factors.append(" Just Below $10K CTR Threshold")
        
        # Protective factors
        if is_bank_800:
            risk_factors.append(" Bank 800 (Trusted - Protective)")
        if is_night:
            risk_factors.append(" Night Transaction (0.53x - Protective)")
        
        if risk_factors:
            for factor in risk_factors:
                if "Protective" in factor:
                    st.success(factor)
                else:
                    st.warning(factor)
        else:
            st.info("No significant risk factors identified")

st.markdown("---")

# Prediction History Table
st.markdown("### Prediction History")

if len(st.session_state.prediction_history) > 0:
    # Convert to DataFrame
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_predictions = len(history_df)
    high_risk_count = len(history_df[history_df['Status'].str.contains('HIGH RISK')])
    low_risk_count = total_predictions - high_risk_count
    high_risk_pct = (high_risk_count / total_predictions * 100) if total_predictions > 0 else 0
    
    with col1:
        st.metric("Total Scored", total_predictions)
    with col2:
        st.metric("High Risk", high_risk_count, delta=f"{high_risk_pct:.1f}%")
    with col3:
        st.metric("Low Risk", low_risk_count)
    with col4:
        # Clear history button
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    
    st.markdown("---")
    
    # Display table with most recent first
    st.dataframe(
        history_df.iloc[::-1],  # Reverse to show most recent first
        use_container_width=True,
        height=400,
        column_config={
            "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
            "Sender Bank": st.column_config.NumberColumn("Sender Bank", format="%d"),
            "Receiver Bank": st.column_config.NumberColumn("Receiver Bank", format="%d"),
            "Amount": st.column_config.TextColumn("Amount", width="small"),
            "Currency": st.column_config.TextColumn("Currency", width="small"),
            "Payment Format": st.column_config.TextColumn("Payment Format", width="small"),
            "Risk Score": st.column_config.TextColumn("Risk Score", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium")
        },
        hide_index=True
    )
    
    # Download option
    csv = history_df.to_csv(index=False)
    st.download_button(
        label="Download History as CSV",
        data=csv,
        file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.info("No predictions yet. Score a transaction above to start building your history.")
