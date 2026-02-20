import numpy as np
import pytest
from sklearn.tree import PCTClassifier


def expected_clus_gini_sum(y):
    """
    Expected ClusGini after modification:
    - sum over outputs (not averaged)
    - per-output denominator (ignores missing via NaN)
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    tot = 0.0
    n_outputs = y.shape[1]
    for o in range(n_outputs):
        col = y[:, o]
        col = col[~np.isnan(col)].astype(int)
        if col.size == 0:
            continue
        counts = np.bincount(col).astype(float)
        p = counts / counts.sum()
        tot += 1.0 - np.sum(p * p)
    return tot


def test_clus_gini_root_matches_per_output_totals():
    X = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)

    y = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ], dtype=float)

    clf = PCTClassifier(
        criterion="clus_gini",
        max_depth=1,
        random_state=0,
        missing_target_attr_handling="parent_node",
    ).fit(X, y)

    got = float(clf.tree_.impurity[0])
    exp = expected_clus_gini_sum(y)

    assert got == pytest.approx(exp, rel=1e-12, abs=1e-12)


def test_clus_gini_handles_missing_targets_per_output():
    X = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)

    y_missing = np.array([
        [0.0, 0.0],
        [0.0, np.nan],
        [1.0, 1.0],
        [np.nan, 1.0],
    ], dtype=float)

    clf = PCTClassifier(
        criterion="clus_gini",
        max_depth=1,
        random_state=0,
        missing_target_attr_handling="parent_node",
    ).fit(X, y_missing)

    got = float(clf.tree_.impurity[0])
    exp = expected_clus_gini_sum(y_missing)

    assert got == pytest.approx(exp, rel=1e-12, abs=1e-12)
