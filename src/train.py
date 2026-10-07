"""Train and compare the candidate models, then save the best one.

Usage:  python -m src.train
"""
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import cross_val_score

from .data import split
from .models import candidates

MODEL_PATH = Path("models/best_model.joblib")
REPORT_PATH = Path("models/report.json")


def evaluate(seed: int = 42) -> dict:
    X_train, X_test, y_train, y_test = split(seed=seed)
    results = {}
    for name, model in candidates(seed).items():
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "cv_f1_mean": round(float(cv.mean()), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "test_f1": round(float(f1_score(y_test, pred)), 4),
            # recall of the malignant class (0): missing a malignant case is the costly error
            "test_malignant_recall": round(float(recall_score(y_test, pred, pos_label=0)), 4),
        }
    return results


def main() -> None:
    results = evaluate()
    best = max(results, key=lambda n: results[n]["cv_f1_mean"])
    X_train, _, y_train, _ = split()
    model = candidates()[best].fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps({"best": best, "results": results}, indent=2))
    print(f"{'model':<22}{'cv_f1':>8}{'acc':>8}{'f1':>8}{'recall_M':>10}")
    for name, r in results.items():
        mark = " *" if name == best else ""
        print(f"{name:<22}{r['cv_f1_mean']:>8}{r['test_accuracy']:>8}{r['test_f1']:>8}{r['test_malignant_recall']:>10}{mark}")
    print(f"\nBest model: {best} -> saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
