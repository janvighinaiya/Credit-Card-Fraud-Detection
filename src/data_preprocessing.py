"""
data_preprocessing.py
----------------------
Loading, cleaning, scaling and splitting the transaction data, plus
SMOTE-based resampling to handle the severe class imbalance.
"""

import os

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config


def load_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw transactions CSV. Raises a clear error if it's missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No dataset found at {path}.\n"
            f"Run `python data/generate_synthetic_data.py` first, or place a "
            f"real creditcard.csv at that path."
        )
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop exact duplicates, drop rows with missing labels."""
    df = df.drop_duplicates()
    df = df.dropna(subset=["Class"])
    df["Class"] = df["Class"].astype(int)
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, save_scaler: bool = True):
    """Scale 'Time' and 'Amount' (the PCA components are already roughly
    standardized in the real dataset, so we only need to scale these two)."""
    scaler = StandardScaler()
    cols_to_scale = [c for c in ["Time", "Amount"] if c in X_train.columns]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    if save_scaler:
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, config.SCALER_PATH)

    return X_train, X_test, scaler


def resample_with_smote(X_train, y_train):
    """Oversample the minority (fraud) class in the training set only.
    Never resample the test set -- it must reflect real-world distribution."""
    smote = SMOTE(
        sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
        random_state=config.RANDOM_STATE,
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def prepare_data(path: str = config.RAW_DATA_PATH, use_smote: bool = config.USE_SMOTE):
    """Full preprocessing pipeline: load -> clean -> split -> scale -> (SMOTE)."""
    df = load_data(path)
    df = clean_data(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,  # preserve the fraud/legit ratio in both splits
    )

    X_train, X_test, _ = scale_features(X_train, X_test)

    if use_smote:
        X_train, y_train = resample_with_smote(X_train, y_train)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Train set: {X_train.shape}, fraud ratio: {y_train.mean():.4f}")
    print(f"Test set:  {X_test.shape}, fraud ratio: {y_test.mean():.4f}")
