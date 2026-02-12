import numpy as np
from sklearn.tree import PCTClassifier

def test_missing_targets_change_root_impurity_counts():
    X = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)

    y_full = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ], dtype=float)

    y_missing = y_full.copy()
    y_missing[1, 1] = np.nan
    y_missing[3, 0] = np.nan

    clf_full = PCTClassifier(
        criterion="clus_entropy",
        max_depth=1,
        random_state=0,
        missing_target_attr_handling="parent_node",
    ).fit(X, y_full)

    clf_miss = PCTClassifier(
        criterion="clus_entropy",
        max_depth=1,
        random_state=0,
        missing_target_attr_handling="parent_node",
    ).fit(X, y_missing)

    assert float(clf_full.tree_.impurity[0]) != float(clf_miss.tree_.impurity[0])
