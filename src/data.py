"""Data loading helpers for the Breast Cancer Wisconsin dataset (bundled with scikit-learn)."""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_dataframe() -> pd.DataFrame:
    """Return the dataset as a DataFrame with a `target` column (1 = benign, 0 = malignant)."""
    raw = load_breast_cancer(as_frame=True)
    return raw.frame


def split(test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split."""
    df = load_dataframe()
    X, y = df.drop(columns="target"), df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
