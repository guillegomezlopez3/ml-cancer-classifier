# ml-cancer-classifier

Supervised-learning mini project: compare three scikit-learn classifiers on the **Breast Cancer Wisconsin** dataset (569 samples, 30 numeric features) and keep the best one.

## What it does

- Loads the dataset bundled with scikit-learn (no downloads needed)
- Builds three pipelines: Logistic Regression, k-NN (both with `StandardScaler`) and Random Forest
- Evaluates each with 5-fold cross-validation (F1) and on a stratified hold-out test set
- Reports accuracy, F1 and **recall on the malignant class**, the error that matters most in a medical setting
- Saves the best model with `joblib` and a JSON report
- Small CLI to predict on a dataset row

## Example result

```
model                    cv_f1     acc      f1  recall_M
logistic_regression     0.9843  0.9825  0.9861    0.9762 *
knn                     0.9775  0.9737  0.9796    0.9286
random_forest           0.9684  0.9561  0.9655    0.9286
```

Logistic Regression wins here. On a small, well-scaled tabular dataset a simple linear model is hard to beat, and it is easy to explain.

## Stack

Python 3.11+ · scikit-learn · pandas · joblib · pytest

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.train            # compare models and save the best one
python -m src.predict --row 3  # predict on one sample
pytest                         # run tests
```

## Notes and limits

This is a learning project, not a diagnostic tool. The scaler lives inside the pipeline so test data never leaks into preprocessing. Results come from one split with a fixed seed.
