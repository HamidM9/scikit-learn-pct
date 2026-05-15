import numpy as np

from sklearn.tree import PCTClassifier, PCTRegressor


def test_pct_classifier_basic_fit_predict():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [1, 1], [1, 1]])

    clf = PCTClassifier(
        criterion="clus_gini",
        max_depth=2,
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X)
    assert pred.shape == y.shape


def test_pct_regressor_basic_fit_predict():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([[0.0, 1.0], [0.2, 1.1], [2.0, 3.0], [2.1, 3.1], [5.0, 6.0], [5.1, 6.1]])

    reg = PCTRegressor(
        criterion="svars",
        max_depth=2,
        random_state=0,
    )
    reg.fit(X, y)

    pred = reg.predict(X)
    assert pred.shape == y.shape