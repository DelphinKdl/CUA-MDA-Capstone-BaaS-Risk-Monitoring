# BaaS Risk Monitoring using Synthetic Data

## Project Overview
We aim to build a machine learning-powered fraud detection and monitoring dashboard using IBM Synthetic Datasets for Core Banking and Money Laundering.

## Project Structure
- `app/`: live dashboard code (Streamlit)
- `data/`: raw and processed datasets
- `notebooks/`: Jupyter exploration and model experiments
- `scripts/`: reusable code
- `reports/`: slides and visuals

## Objective
- Detect high-risk transactions (e.g., laundering, check fraud)
- Build an interactive dashboard for transaction monitoring
- Demonstrate real-time fraud flagging capability

## Tech Stack
Python, Scikit-learn, Pandas, Streamlit, GitHub, AWS

##  Team Members
- Delphin Kaduli
- Tycho Janssen 
- Solomon Pinto.

## Setup
```bash
git https://github.com/DelphinKdl/CUA-MDA-Capstone-BaaS-Risk-Monitoring.git
pip install -r requirements.txt


**Banking-as-a-Service (BaaS) Risk Monitoring**
*Capstone Project Overview | The Catholic University of America*

---

###  Project Summary

We are developing a **machine learning-based monitoring system** to detect and visualize **fraudulent transactions** and **money laundering risks** in **Banking-as-a-Service (BaaS)** environments. With the rise of embedded finance, BaaS providers are increasingly vulnerable to illicit activity due to the wide reach of their FinTech clients. Our system aims to close this monitoring gap.

---

###  Problem Statement

**BaaS platforms** often serve hundreds of downstream FinTechs. These clients introduce new user bases, transaction behaviors, and risk patterns. Traditional compliance systems often fail to:

* Flag suspicious activity across layered client structures.
* Visualize patterns indicative of structuring or mule activity.
* Provide actionable insights in near real-time.

---

### Our Approach

We aim to build a **machine learning pipeline and interactive dashboard** that:

1. **Processes transactional data** to detect:

   * Structuring patterns
   * Unusual velocity or frequency
   * Use of proxy accounts

2. **Generates risk scores** using classification models (e.g., Random Forest, XGBoost)

3. **Feeds alerts and visualizations** into a secure, private **Streamlit dashboard** hosted on **AWS EC2**.

4. **Simulates batch streaming** to replicate real-time behavior with synthetic or anonymized data.

---

###  Deployment & Data Privacy

* **Private deployment** on AWS EC2 for faculty & mentor review.
* **No public exposure** of sensitive data or metadata.
* **.gitignore** includes dataset & Kaggle credentials to prevent leakage.
* Real-world design supports daily/real-time ingestion.

---

###  Update Frequency (Prototype vs Real World)

| Environment | Update Cycle                | Purpose                            |
| ----------- | --------------------------- | ---------------------------------- |
| Prototype   | Simulated daily batches     | Model development, visualization   |
| Real-world  | Real-time + Daily summaries | Compliance alerting, risk insights |

---

###  Visual Output

* Time-series charts of flagged transaction volumes
* Risk heatmaps by customer segment or FinTech partner
* Account-level drill-downs
* Model decision rationale (SHAP values)

---

### 🪄 Request to IBM

We are using the publicly available IBM dataset for AML detection from Kaggle. However, we kindly request:

* **Assistance in verifying our usage scenario is aligned with the dataset terms**
* **Access to metadata fields (if permissible) or simulated streaming batches** for stronger real-time modeling
* **Feedback from IBM researchers or practitioners** on fraud detection in embedded finance

---

### Why This Matters

BaaS ecosystems are a growing target for illicit finance due to their scale and fragmentation. A proactive, ML-powered risk dashboard can:

* Reduce compliance gaps
* Help providers protect reputation
* Assist regulators and internal auditors

---

###  Contact & Next Steps

We are happy to:

* Share our GitHub repo and deployment link with IBM under NDA or private access
* Present our methodology and findings to IBM teams
* Explore potential collaborations for productization or further research

> *Prepared by: Delphin Kdl & team*
> *Catholic University of America, Data Analytics Capstone*
