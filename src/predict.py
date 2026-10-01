"""
predict.py
-----------
Load a trained model + scaler and score new, unseen transactions.

Usage:
    from src.predict import FraudPredictor

    predictor = FraudPredictor()
    result = predictor.predict({
        "Time": 50000, "V1": -1.2, "V2": 0.5, ... , "V28": 0.03, "Amount": 149.62
    })
"""

import joblib
import pandas as pd

from src import config


class FraudPredictor:
    def __init__(self, model_path: str = config.BEST_MODEL_PATH, scaler_path: str = config.SCALER_PATH):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def _prepare_input(self, transaction: dict) -> pd.DataFrame:
        df = pd.DataFrame([transaction])
        cols_to_scale = [c for c in ["Time", "Amount"] if c in df.columns]
        df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        return df

    def predict(self, transaction: dict, threshold: float = config.CLASSIFICATION_THRESHOLD) -> dict:
        """
        transaction: dict with keys Time, V1..V28, Amount (same schema as training data).
        Returns: {"is_fraud": bool, "fraud_probability": float}
        """
        X = self._prepare_input(transaction)
        # Align column order to what the model was trained on
        X = X[self.model.feature_names_in_] if hasattr(self.model, "feature_names_in_") else X
        proba = self.model.predict_proba(X)[0, 1]
        return {
            "is_fraud": bool(proba >= threshold),
            "fraud_probability": round(float(proba), 6),
        }

    def predict_batch(self, transactions: pd.DataFrame, threshold: float = config.CLASSIFICATION_THRESHOLD) -> pd.DataFrame:
        """Score a whole DataFrame of transactions at once."""
        df = transactions.copy()
        cols_to_scale = [c for c in ["Time", "Amount"] if c in df.columns]
        df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

        proba = self.model.predict_proba(df)[:, 1]
        out = transactions.copy()
        out["fraud_probability"] = proba
        out["is_fraud"] = proba >= threshold
        return out


if __name__ == "__main__":
    # Quick smoke test using a random row from the raw dataset, if available.
    import os

    if os.path.exists(config.RAW_DATA_PATH) and os.path.exists(config.BEST_MODEL_PATH):
        df = pd.read_csv(config.RAW_DATA_PATH).drop(columns=["Class"]).sample(1, random_state=1)
        predictor = FraudPredictor()
        sample = df.iloc[0].to_dict()
        print("Sample transaction:", sample)
        print("Prediction:", predictor.predict(sample))
    else:
        print("Train a model first: python main.py")
