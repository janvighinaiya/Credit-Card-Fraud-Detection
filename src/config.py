"""
config.py
---------
Central place for paths and hyperparameters so nothing is hardcoded
throughout the pipeline.
"""

import os

# --- Paths -----------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "creditcard.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# --- Data split --------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Class imbalance handling -----------------------------------------------
USE_SMOTE = True
SMOTE_SAMPLING_STRATEGY = 0.1  # ratio of minority:majority after resampling

# --- Model hyperparameters ---------------------------------------------------
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Decision threshold for classifying a transaction as fraud (probability).
# Tune this based on the precision/recall trade-off your use case needs.
CLASSIFICATION_THRESHOLD = 0.5
