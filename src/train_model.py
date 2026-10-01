"""
train_model.py
----------------
Trains Logistic Regression, Random Forest and XGBoost classifiers,
compares them on PR-AUC (the most meaningful metric under heavy class
imbalance), and saves the best-performing model to disk.
"""

import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src import config
from src.data_preprocessing import prepare_data


def get_models():
    """Instantiate all candidate models with their configured hyperparameters."""
    return {
        "logistic_regression": LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS),
        "random_forest": RandomForestClassifier(**config.RANDOM_FOREST_PARAMS),
        "xgboost": XGBClassifier(**config.XGBOOST_PARAMS),
    }


def train_all_models(X_train, y_train):
    """Fit every candidate model and return them in a dict."""
    models = get_models()
    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def select_best_model(trained_models: dict, X_test, y_test):
    """Pick the model with the highest PR-AUC (average precision) on the
    held-out test set -- more informative than accuracy for imbalanced data."""
    scores = {}
    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, y_proba)
        scores[name] = pr_auc
        print(f"{name:>22s}  PR-AUC = {pr_auc:.4f}")

    best_name = max(scores, key=scores.get)
    print(f"\nBest model: {best_name} (PR-AUC = {scores[best_name]:.4f})")
    return best_name, trained_models[best_name], scores


def save_model(model, path: str = config.BEST_MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model -> {path}")


def run_training_pipeline():
    X_train, X_test, y_train, y_test = prepare_data()
    trained_models = train_all_models(X_train, y_train)
    best_name, best_model, scores = select_best_model(trained_models, X_test, y_test)
    save_model(best_model)
    return best_name, best_model, scores, (X_test, y_test)


if __name__ == "__main__":
    run_training_pipeline()
