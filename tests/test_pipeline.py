from src.data import load_dataframe, split
from src.models import candidates


def test_dataset_shape():
    df = load_dataframe()
    assert df.shape == (569, 31)
    assert set(df["target"].unique()) == {0, 1}


def test_split_is_stratified_and_disjoint():
    X_train, X_test, y_train, y_test = split()
    assert len(X_train) + len(X_test) == 569
    assert abs(y_train.mean() - y_test.mean()) < 0.03
    assert set(X_train.index).isdisjoint(X_test.index)


def test_all_models_reach_reasonable_accuracy():
    X_train, X_test, y_train, y_test = split()
    for name, model in candidates().items():
        model.fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.9, name
