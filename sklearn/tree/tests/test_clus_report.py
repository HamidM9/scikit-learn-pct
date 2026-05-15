import numpy as np

from sklearn.tree import PCTClassifier, PCTRegressor
from sklearn.tree._clus_report import clus_report


def test_clus_report_classifier_runs():
    X = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = PCTClassifier(
        criterion="clus_gini",
        max_depth=1,
        random_state=0,
    )
    clf.fit(X, y)

    report = clus_report(clf, X, y)

    assert "model" in report
    assert "classification" in report
    assert report["model"]["nodes"] >= 1
    assert "accuracy" in report["classification"]
    assert "hamming_loss" in report["classification"]


def test_clus_report_regressor_runs():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    reg = PCTRegressor(
        criterion="svars",
        max_depth=1,
        random_state=0,
    )
    reg.fit(X, y)

    report = clus_report(reg, X, y)

    assert "model" in report
    assert "regression" in report
    assert report["model"]["nodes"] >= 1
    assert "mse" in report["regression"]
    assert "rmse" in report["regression"]