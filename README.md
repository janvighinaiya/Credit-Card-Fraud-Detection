# 💳 Credit Card Fraud Detection using Machine Learning

A production-style, end-to-end machine learning project that detects fraudulent credit card transactions using Python and scikit-learn. Built with a clean, modular codebase suitable for a GitHub portfolio.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

Credit card fraud detection is a classic **imbalanced classification** problem — fraudulent transactions typically make up less than 1% of all transactions. This project demonstrates a complete ML pipeline:

- Synthetic transaction data generation (mirrors the structure of the popular Kaggle "Credit Card Fraud Detection" dataset — V1–V28 PCA features, `Amount`, `Time`, `Class`)
- Data preprocessing & feature scaling
- Handling class imbalance with **SMOTE** (Synthetic Minority Oversampling)
- Training multiple models: **Logistic Regression, Random Forest, XGBoost**
- Evaluation with metrics that actually matter for fraud (Precision, Recall, F1, ROC-AUC, PR-AUC — **not just accuracy**)
- Model persistence and a reusable prediction script
- Clean, modular, well-documented code

## 🗂️ Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   ├── generate_synthetic_data.py   # Creates a realistic synthetic dataset
│   └── raw/                         # Dataset lives here (creditcard.csv)
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Central configuration (paths, params)
│   ├── data_preprocessing.py        # Cleaning, scaling, train/test split, SMOTE
│   ├── train_model.py               # Trains & saves models
│   ├── evaluate_model.py            # Metrics, confusion matrix, ROC/PR curves
│   └── predict.py                   # Load a saved model & score new transactions
│
├── models/                          # Saved trained models (.pkl) go here
├── notebooks/
│   └── eda.ipynb                    # Exploratory Data Analysis (optional)
├── tests/
│   └── test_preprocessing.py        # Basic unit tests
│
├── main.py                          # Single entry point: run full pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate data (or drop your own creditcard.csv into data/raw/)
python data/generate_synthetic_data.py

# 5. Run the full pipeline (preprocess -> train -> evaluate)
python main.py
```

## 📊 Using the Real Dataset (Recommended for a Portfolio)

This repo ships with a **synthetic data generator** so the pipeline runs out of the box with no downloads. For a real-world result to showcase, download the actual Kaggle dataset and drop it in:

1. Go to [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it at `data/raw/creditcard.csv`
4. Re-run `python main.py` — the pipeline automatically uses the real file if present, and falls back to synthetic data otherwise.

## 🧠 Models Included

| Model | Why it's included |
|---|---|
| Logistic Regression | Fast, interpretable baseline |
| Random Forest | Strong non-linear baseline, handles imbalance reasonably well |
| XGBoost | Typically the best performer on this kind of tabular, imbalanced data |

The best model (by PR-AUC) is automatically saved to `models/best_model.pkl`.

## 📈 Why Not Just "Accuracy"?

With ~99.8% of transactions being legitimate, a model that predicts "not fraud" every time scores 99.8% accuracy — and is completely useless. This project reports:

- **Precision** — of the transactions flagged as fraud, how many actually were?
- **Recall** — of all actual fraud, how much did we catch?
- **F1-score** — balance of precision & recall
- **ROC-AUC** and **PR-AUC** (PR-AUC is the more meaningful metric under heavy class imbalance)
- **Confusion matrix**

## 🔮 Making Predictions

```python
from src.predict import FraudPredictor

predictor = FraudPredictor(model_path="models/best_model.pkl")
result = predictor.predict(transaction_dict)
print(result)  # {'is_fraud': False, 'fraud_probability': 0.0123}
```

## 🛠️ Tech Stack

- Python 3.9+
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- xgboost
- matplotlib, seaborn
- joblib

## 📄 License

MIT License — free to use for learning and portfolio purposes.

## 🙋 Author

Add your name, LinkedIn, and portfolio link here before publishing.
