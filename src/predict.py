"""Predict from the saved model using a sample row of the dataset.

Usage:  python -m src.predict --row 0
"""
import argparse

import joblib

from .data import load_dataframe
from .train import MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict on a row of the dataset")
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()
    df = load_dataframe()
    sample = df.drop(columns="target").iloc[[args.row]]
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(sample)[0]
    label = "benign" if proba[1] >= 0.5 else "malignant"
    print(f"Row {args.row}: {label} (p_benign={proba[1]:.3f}) | actual: "
          f"{'benign' if df.iloc[args.row]['target'] == 1 else 'malignant'}")


if __name__ == "__main__":
    main()
