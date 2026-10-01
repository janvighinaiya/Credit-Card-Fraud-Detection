"""
test_preprocessing.py
-----------------------
Basic sanity checks for the preprocessing pipeline.
Run with: pytest tests/
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_synthetic_data import generate_synthetic_data
from src.data_preprocessing import clean_data, split_features_target


def test_generate_synthetic_data_shape():
    df = generate_synthetic_data(n_samples=1000, fraud_ratio=0.05, random_state=1)
    assert len(df) == 1000
    assert "Class" in df.columns
    assert set(df["Class"].unique()).issubset({0, 1})


def test_fraud_ratio_roughly_correct():
    df = generate_synthetic_data(n_samples=5000, fraud_ratio=0.05, random_state=1)
    ratio = df["Class"].mean()
    assert 0.03 < ratio < 0.08


def test_clean_data_removes_duplicates():
    df = pd.DataFrame({"Time": [1, 1], "Amount": [10, 10], "Class": [0, 0]})
    cleaned = clean_data(df)
    assert len(cleaned) == 1


def test_split_features_target():
    df = generate_synthetic_data(n_samples=200, fraud_ratio=0.1, random_state=1)
    X, y = split_features_target(df)
    assert "Class" not in X.columns
    assert len(X) == len(y)
