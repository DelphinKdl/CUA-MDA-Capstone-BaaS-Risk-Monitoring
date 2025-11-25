# Anti-Money Laundering Detection System
**Production-Ready AML Detection for Banking Compliance**

## Project Overview
A machine learning-powered Anti-Money Laundering (AML) detection system with an interactive Streamlit dashboard. Built using IBM's Synthetic AML Dataset (31.9M transactions), this system achieves **73% precision** and ***ROC-AUC 97.8%**, making it production-ready for banking compliance teams.

##  Live Demo
 **[Visit Live Dashboard](https://cua-mda-aml-prediction.streamlit.app/)**

##  Key Features
- **Real-time Transaction Scoring**: Instant risk assessment for individual transactions
- **Calibrated Risk Scores**: Reliable probability estimates (when model says 10% risk, it means 10%)
- **Automatic Feature Engineering**: Extracts 20+ risk indicators from raw transaction data
- **Production-Optimized**: 27x reduction in false positives vs baseline models
- **Interactive Dashboard**: User-friendly interface for compliance analysts

## Model Performance
- **Precision**: 72.75% (73% of alerts are true laundering cases)
- **Recall**: 18.87% (catches 1 in 5 laundering cases)
- **ROC-AUC**: 97.81% (excellent discrimination)
- **False Positive Rate**: 0.01% (only 752 false alarms out of 6.4M normal transactions)
- **Daily Alert Volume**: ~860 alerts (manageable workload)

##  Project Structure
```
├── app/                    # Streamlit dashboard
│   ├── Home.py            # Main page with KPIs and model overview
│   ├── pages/
│   │   ├── 01_Model_Validation.py      # Performance metrics & confusion matrix
│   │   ├── 02_Investigator_Workbench.py # Real-time transaction scoring
│   │   └── 04_Data_Insight.py          # EDA visualizations
├── models/                 # Trained model artifacts
│   ├── calibrated_lightgbm_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   └── model_config.json
├── note/              # Jupyter notebooks for model development
├── data/                   # find it on IBM website
└── requirements.txt        # Python dependencies
```

## Tech Stack
- **ML Framework**: LightGBM with Isotonic Calibration
- **Dashboard**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Deployment**: [Your deployment platform]

##  Dataset
- **Source**: IBM Synthetic AML Dataset
- **Size**: 31,898,238 transactions
- **Period**: September 1-28, 2022
- **Laundering Cases**: 35,230 (0.11%)
- **Class Imbalance**: 1:905 ratio

##  Team Members
**Catholic University of America - MDA Capstone Project Fall 2025**
- Delphin Kaduli
- Tycho Janssen
- Solomon Pinto

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
# Clone the repository
git clone https://github.com/DelphinKdl/CUA-MDA-Capstone-BaaS-Risk-Monitoring.git
cd CUA-MDA-Capstone-BaaS-Risk-Monitoring

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app/Home.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Dashboard Pages

### 1. Home
- Key performance indicators (KPIs)
- Champion model overview
- Business impact metrics

### 2. Model Validation
- Confusion matrix visualization
- Performance metrics
- Model configuration details

### 3. Investigator Workbench
- **Star Feature**: Real-time transaction risk scoring
- User-friendly input form
- Automatic feature engineering
- Instant risk assessment with recommendations

### 4. Data Insights
- Dataset overview and statistics
- Class distribution analysis
- Payment format patterns
- Temporal patterns (day of week)
- Structuring detection analysis

## Key Risk Indicators Detected
- **ACH Payment Format**: 49x baseline risk
- **Weekend Transactions**: 3x baseline risk
- **Structuring Pattern**: $9K-$10K range (8.3x risk)
- **High-Risk Institutions**: Bank-level risk patterns
- **Currency Risk Patterns**: UK Pound structuring
- **Temporal Patterns**: Day of week and hour analysis

## Model Details
- **Algorithm**: LightGBM with Probability Calibration
- **Training Data**: 25.5M transactions (80%)
- **Test Data**: 6.4M transactions (20%)
- **Features**: 20 engineered features
- **Threshold**: 0.10 (optimized for precision)
- **Training Strategy**: Full dataset (no undersampling)

## Business Impact
- **62% reduction** in analyst workload vs alternative models
- **27x reduction** in false positives
- **73% confidence** in every alert
- **Manageable alert volume**: ~860 alerts per day
- **Precision-focused strategy**: Maximizes analyst efficiency

## Acknowledgments
- IBM for providing the Synthetic AML Dataset
- Catholic University of America MDA Program
- Project advisors: Dr. Matthew Jacob

## License
MIT License

Copyright (c) 2025 Delphin Kaduli, Tycho Janssen and Solomon Pinto.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights

