import numpy as np
import pytest

from sklearn.tree import PCTClassifier


def test_pct_ssl_classification_weight_one_ignores_unlabeled_rows():
    rng = np.random.RandomState(11)

    X = rng.normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int).reshape(-1, 1)

    y_ssl = y.astype(float).copy()
    y_ssl[::4, 0] = np.nan
    keep = ~np.isnan(y_ssl[:, 0])

    est_full = PCTClassifier(
        criterion="clus_gini",
        random_state=0,
        ssl=True,
        ssl_method="clus_pct",
        ssl_weight=1.0,
        RemoveMissingTarget="No",
        compat_mode="clus_v1",
        tie_break="clus",
        split_position="clus_exact",
    )

    est_labeled = PCTClassifier(
        criterion="clus_gini",
        random_state=0,
        ssl=False,
        compat_mode="clus_v1",
        tie_break="clus",
        split_position="clus_exact",
    )

    est_full.fit(X, y_ssl)
    est_labeled.fit(X[keep], y[keep])

    assert np.all(est_full.predict(X) == est_labeled.predict(X))


def test_pct_ssl_classification_rejects_missing_targets_without_ssl():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array([[0.0], [0.0], [np.nan], [1.0]])

    est = PCTClassifier(
        criterion="clus_gini",
        random_state=0,
        ssl=False,
        RemoveMissingTarget="No",
        compat_mode="clus_v1",
    )

    with pytest.raises(ValueError, match="Missing|missing|NaN|unlabeled"):
        est.fit(X, y)


def test_pct_ssl_classification_accepts_missing_targets_with_ssl():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )
    y = np.array([[0.0], [0.0], [np.nan], [np.nan], [1.0], [1.0]])

    est = PCTClassifier(
        criterion="clus_gini",
        random_state=0,
        ssl=True,
        ssl_method="clus_pct",
        ssl_weight=0.5,
        RemoveMissingTarget="No",
        compat_mode="clus_v1",
        tie_break="clus",
        split_position="clus_exact",
    )

    est.fit(X, y)

    pred = est.predict(X)

    assert pred.shape == (X.shape[0],)
    assert set(np.unique(pred)).issubset({0, 1})


def test_pct_ssl_classification_weight_grid_sets_selected_weight():
    rng = np.random.RandomState(13)

    X = rng.normal(size=(50, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1).astype(float)

    y[::5, 0] = np.nan

    est = PCTClassifier(
        criterion="clus_gini",
        random_state=0,
        ssl=True,
        ssl_method="clus_pct",
        ssl_possible_weights=[0.0, 0.5, 1.0],
        ssl_internal_folds=3,
        RemoveMissingTarget="No",
        compat_mode="clus_v1",
        tie_break="clus",
        split_position="clus_exact",
    )

    est.fit(X, y)

    assert hasattr(est, "ssl_weight_")
    assert est.ssl_weight_ in [0.0, 0.5, 1.0]


def test_pct_ssl_classification_invalid_weight_raises():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([[0.0], [0.0], [np.nan], [1.0]])

    est = PCTClassifier(
        criterion="clus_gini",
        random_state=0,
        ssl=True,
        ssl_method="clus_pct",
        ssl_weight=1.5,
        RemoveMissingTarget="No",
        compat_mode="clus_v1",
    )

    with pytest.raises(ValueError, match="ssl_weight"):
        est.fit(X, y)