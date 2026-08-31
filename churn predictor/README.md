# Salesforce Churn Predictor

**Real business framing, imbalanced data**

## Introduction

Taking a telecom churn dataset, engineer features from usage patterns and account history, and build a binary classifier. The core challenge is learning to optimize for recall vs precision based on the real business cost of missing a churning customer.

## Tech Stack

- Python, pandas, scikit-learn, imbalanced-learn, Matplotlib/Seaborn, Streamlit

## Key Highlights
- Data overview
<img width="1368" height="742" alt="Screenshot 2026-08-31 at 1 45 33 PM" src="https://github.com/user-attachments/assets/02d8e1c0-6077-4042-89d7-0c8a7a0050e9" />
  <img width="1384" height="789" alt="Screenshot 2026-08-31 at 1 45 58 PM" src="https://github.com/user-attachments/assets/c606de26-19ff-469f-a228-c758f28a445e" />
- Feature engineering from raw transactional and usage data(random forest)
<img width="1295" height="796" alt="Screenshot 2026-08-31 at 1 57 43 PM" src="https://github.com/user-attachments/assets/dc51bc4c-1503-46c1-8c1a-cc71f94a80b6" />

- Why accuracy is the wrong metric for imbalanced churn datasets
   <img width="1313" height="710" alt="Screenshot 2026-08-31 at 1 47 50 PM" src="https://github.com/user-attachments/assets/c5eb15f8-eda0-411d-af7b-6ead57dc47db" />
  
- Decision threshold tuning and the precision-recall tradeoff
   <img width="1359" height="709" alt="Screenshot 2026-08-31 at 1 57 04 PM" src="https://github.com/user-attachments/assets/b97d2a6b-a3c7-4422-8bee-ef16b2e3addc" />

- Handling imbalanced data with SMOTE
<img width="1316" height="662" alt="Screenshot 2026-08-31 at 1 55 03 PM" src="https://github.com/user-attachments/assets/7e7007cd-2912-43ce-9c8f-4742a8e4e1bd" />

- Building an interactive prediction app in Streamlit
  <img width="1338" height="793" alt="Screenshot 2026-08-31 at 1 53 34 PM" src="https://github.com/user-attachments/assets/7b075252-5b44-42b7-8456-92e347cb9c37" />

<img width="1357" height="263" alt="Screenshot 2026-08-31 at 1 53 52 PM" src="https://github.com/user-attachments/assets/1bcdbc0e-60ff-4601-9001-46db1e56ace2" />



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
