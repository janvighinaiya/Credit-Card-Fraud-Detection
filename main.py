"""
main.py
--------
Single entry point for the full fraud detection pipeline:

    1. Load data (generate synthetic data automatically if none exists)
    2. Preprocess (clean, split, scale, SMOTE)
    3. Train Logistic Regression, Random Forest and XGBoost
    4. Select the best model by PR-AUC
    5. Evaluate it in detail and save plots
    6. Persist the best model + scaler to models/

Run:
    python main.py
"""

import os

from src import config
from src.data_preprocessing import prepare_data
from src.evaluate_model import evaluate_model, generate_all_plots
from src.train_model import run_training_pipeline


def ensure_data_exists():
    if not os.path.exists(config.RAW_DATA_PATH):
        print("No dataset found -- generating synthetic data...")
        from data.generate_synthetic_data import generate_synthetic_data

        df = generate_synthetic_data(n_samples=50_000, fraud_ratio=0.0017)
        os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
        df.to_csv(config.RAW_DATA_PATH, index=False)
        print(f"Synthetic dataset saved -> {config.RAW_DATA_PATH}\n")


def main():
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION -- FULL PIPELINE")
    print("=" * 60)

    ensure_data_exists()

    print("\n[1/3] Training candidate models...")
    best_name, best_model, scores, (X_test, y_test) = run_training_pipeline()

    print(f"\n[2/3] Evaluating best model: {best_name}")
    results = evaluate_model(best_model, X_test, y_test)

    print("\n[3/3] Generating evaluation plots...")
    generate_all_plots(results, y_test)

    print("\nDone. Best model saved to:", config.BEST_MODEL_PATH)
    print("Plots saved to: models/plots/")


if __name__ == "__main__":
    main()
