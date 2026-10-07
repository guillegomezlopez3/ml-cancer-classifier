"""Candidate models, all wrapped in pipelines so scaling is learned only from training data."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def candidates(seed: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline([("scale", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=1000))]),
        "knn": Pipeline([("scale", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=7))]),
        "random_forest": Pipeline([("clf", RandomForestClassifier(n_estimators=200, random_state=seed))]),
    }
