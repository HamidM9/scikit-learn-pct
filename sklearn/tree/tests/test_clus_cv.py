import numpy as np

from sklearn.tree import PCTClassifier
from sklearn.tree._clus_cv import ClusXValSelection, clus_cross_validate


def test_clus_xval_selection_is_deterministic():
    X = np.arange(20).reshape(-1, 1)

    sel1 = ClusXValSelection(n_splits=5, random_state=0)
    sel2 = ClusXValSelection(n_splits=5, random_state=0)

    np.testing.assert_array_equal(
        sel1.make_folds(X.shape[0]),
        sel2.make_folds(X.shape[0]),
    )


def test_clus_xval_selection_covers_all_samples_once_as_test():
    X = np.arange(20).reshape(-1, 1)

    sel = ClusXValSelection(n_splits=5, random_state=0)

    test_seen = []
    for _, test_idx in sel.split(X):
        test_seen.extend(test_idx.tolist())

    assert sorted(test_seen) == list(range(20))


def test_clus_cross_validate_pct_classifier_runs():
    X = np.array([
        [0.0],
        [0.1],
        [0.2],
        [0.8],
        [0.9],
        [1.0],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    clf = PCTClassifier(
        criterion="clus_gini",
        max_depth=1,
        random_state=0,
    )

    result = clus_cross_validate(
        clf,
        X,
        y,
        n_splits=3,
        random_state=0,
    )

    assert result["scores"].shape == (3,)
    assert len(result["estimators"]) == 3
    assert len(result["splits"]) == 3