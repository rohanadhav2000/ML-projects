# Salesforce Churn Predictor

**Real business framing, imbalanced data**

## Introduction

Taking a telecom churn dataset, engineer features from usage patterns and account history, and build a binary classifier. The core challenge is learning to optimize for recall vs precision based on the real business cost of missing a churning customer.

## Tech Stack

- Python, pandas, scikit-learn, imbalanced-learn, Matplotlib/Seaborn, Streamlit

## Key Highlights

- Feature engineering from raw transactional and usage data
  <img width="1368" height="742" alt="Screenshot 2026-08-31 at 1 45 33 PM" src="https://github.com/user-attachments/assets/02d8e1c0-6077-4042-89d7-0c8a7a0050e9" />
  <img width="1384" height="789" alt="Screenshot 2026-08-31 at 1 45 58 PM" src="https://github.com/user-attachments/assets/c606de26-19ff-469f-a228-c758f28a445e" />

- Why accuracy is the wrong metric for imbalanced churn datasets
- Decision threshold tuning and the precision-recall tradeoff
- Handling imbalanced data with SMOTE
- Building an interactive prediction app in Streamlit




## Execution

```bash
# From the project root directory
pip install -r requirements.txt
streamlit run 04_churn_predictor/app.py
```

## Data Sources

The app tries these sources in order:

1. **Local CSV** — Place the CSV file at `04_churn_predictor/data/churn.csv`
2. **Kaggle** — Automatically downloads the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset if Kaggle credentials are configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars)
3. **Synthetic data** — Generates ~7 000 realistic rows (~26 % churn rate) as a fallback
