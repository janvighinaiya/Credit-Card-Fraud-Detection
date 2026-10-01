"""
generate_synthetic_data.py
---------------------------
Generates a synthetic credit card transactions dataset that mirrors the
structure of the well-known Kaggle "Credit Card Fraud Detection" dataset:

    Time, V1, V2, ..., V28, Amount, Class

- V1-V28 : PCA-anonymized numerical features (simulated here via Gaussian
           mixtures so fraud and legitimate transactions have different,
           overlapping distributions -- just like the real dataset).
- Time   : Seconds elapsed since the first transaction.
- Amount : Transaction amount.
- Class  : 1 = fraud, 0 = legitimate (heavily imbalanced, ~0.17% fraud).

This lets the whole pipeline run end-to-end with zero downloads. Swap this
file's output for the real creditcard.csv (from Kaggle) at
data/raw/creditcard.csv for real-world results.

Usage:
    python data/generate_synthetic_data.py --n_samples 50000 --fraud_ratio 0.0017
"""

import argparse
import os

import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples: int = 50_000, fraud_ratio: float = 0.0017, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    n_fraud = max(int(n_samples * fraud_ratio), 20)
    n_legit = n_samples - n_fraud

    n_features = 28  # V1 ... V28

    # Legitimate transactions: standard-normal-ish PCA components
    legit_features = rng.normal(loc=0.0, scale=1.0, size=(n_legit, n_features))

    # Fraudulent transactions: shifted mean + higher variance on a subset of
    # components, so the classes are separable but still overlapping (realistic).
    fraud_shift = rng.normal(loc=0.0, scale=1.0, size=n_features)
    fraud_shift[:8] += rng.choice([-1, 1], size=8) * rng.uniform(0.8, 1.8, size=8)
    fraud_features = rng.normal(loc=fraud_shift, scale=1.0, size=(n_fraud, n_features))
    # Add a chunk of "hard" fraud cases that look almost identical to legit
    # transactions, so the model can't achieve a trivially perfect score.
    n_hard = int(n_fraud * 0.35)
    hard_idx = rng.choice(n_fraud, size=n_hard, replace=False)
    fraud_features[hard_idx] = rng.normal(loc=0.0, scale=1.0, size=(n_hard, n_features))

    features = np.vstack([legit_features, fraud_features])
    labels = np.array([0] * n_legit + [1] * n_fraud)

    # Amount: legit transactions mostly small, fraud skews toward certain ranges
    legit_amount = np.round(rng.gamma(shape=2.0, scale=40.0, size=n_legit), 2)
    fraud_amount = np.round(rng.gamma(shape=1.5, scale=180.0, size=n_fraud), 2)
    amount = np.concatenate([legit_amount, fraud_amount])

    # Time: seconds across a 2-day window, fraud slightly more likely at odd hours
    time = rng.integers(low=0, high=172_800, size=n_samples)

    df = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, n_features + 1)])
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = labels

    # Shuffle rows so fraud isn't grouped at the bottom
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic credit card transaction data.")
    parser.add_argument("--n_samples", type=int, default=50_000, help="Total number of transactions.")
    parser.add_argument("--fraud_ratio", type=float, default=0.0017, help="Proportion of transactions that are fraud.")
    parser.add_argument("--output", type=str, default="data/raw/creditcard.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_synthetic_data(args.n_samples, args.fraud_ratio, args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    fraud_count = int(df["Class"].sum())
    print(f"Generated {len(df):,} transactions -> {args.output}")
    print(f"Fraudulent: {fraud_count:,} ({fraud_count / len(df):.4%})")
    print(f"Legitimate: {len(df) - fraud_count:,}")


if __name__ == "__main__":
    main()
