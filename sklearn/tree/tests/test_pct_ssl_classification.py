import numpy as np
import pytest
from numpy.testing import assert_allclose

from sklearn.tree import PCTClassifier


def test_pct_ssl_classification_accepts_mixed_clustering_roles():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)

    est = PCTClassifier(
        random_state=0,
        ssl=True,
        ssl_method="clus_pct",
        ssl_weight=0.5,
        descriptive_features=[0, 1],
        clustering_features=[0, 1, 2],
        target_features=[2],
        RemoveMissingTarget="No",
    ).fit(X, y)

    pred = np.asarray(est.predict(X))
    assert pred.shape == (30,)


def test_pct_ssl_classification_weight_one_matches_supervised_like_baseline():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] > X[:, 1]).astype(float).reshape(-1, 1)

    sup = PCTClassifier(
        random_state=0,
        ssl=False,
        descriptive_features=[0, 1],
        clustering_features=[2],
        target_features=[2],
    ).fit(X, y)

    ssl = PCTClassifier(
        random_state=0,
        ssl=True,
        ssl_method="clus_pct",
        ssl_weight=1.0,
        descriptive_features=[0, 1],
        clustering_features=[0, 1, 2],
        target_features=[2],
        RemoveMissingTarget="No",
    ).fit(X, y)

    assert_allclose(sup.predict(X), ssl.predict(X))


def test_pct_ssl_classification_allows_missing_targets_but_supervised_rejects():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(20, 2))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)
    y[::3, 0] = np.nan

    with pytest.raises(ValueError, match="Missing classification targets found"):
        PCTClassifier(
            random_state=0,
            ssl=False,
            descriptive_features=[0, 1],
            clustering_features=[2],
            target_features=[2],
            RemoveMissingTarget="No",
        ).fit(X, y)

    est = PCTClassifier(
        random_state=0,
        ssl=True,
        ssl_method="clus_pct",
        ssl_weight=0.5,
        descriptive_features=[0, 1],
        clustering_features=[0, 1, 2],
        target_features=[2],
        RemoveMissingTarget="No",
    ).fit(X, y)

    assert hasattr(est, "tree_")


def test_pct_ssl_classification_constructs_clus_style_column_weights():
    rng = np.random.RandomState(3)
    X = rng.normal(size=(10, 2))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)

    est = PCTClassifier(
        random_state=0,
        ssl=True,
        ssl_method="clus_pct",
        ssl_weight=0.25,
        descriptive_features=[0, 1],
        clustering_features=[0, 1, 2],
        target_features=[2],
        RemoveMissingTarget="No",
    ).fit(X, y)

    assert hasattr(est, "_pct_ssl_column_weights_")
    np.testing.assert_allclose(est._pct_ssl_column_weights_, [1.125, 1.125, 0.75])


def test_pct_ssl_classification_selects_weight_from_grid():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)
    y[::4, 0] = np.nan

    est = PCTClassifier(
        random_state=0,
        ssl=True,
        ssl_method="clus_pct",
        ssl_possible_weights=[1.0, 0.5, 0.0],
        ssl_internal_folds=3,
        descriptive_features=[0, 1],
        clustering_features=[0, 1, 2],
        target_features=[2],
        RemoveMissingTarget="No",
    ).fit(X, y)

    assert est.ssl_weight_ in {1.0, 0.5, 0.0}


def test_pct_ssl_classification_weight_grid_validation():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([[0.0], [1.0], [0.0]])

    with pytest.raises(ValueError, match="ssl_possible_weights must lie in \\[0, 1\\]"):
        PCTClassifier(
            random_state=0,
            ssl=True,
            ssl_method="clus_pct",
            ssl_possible_weights=[-0.1, 0.5, 1.0],
            descriptive_features=[0],
            clustering_features=[0, 1],
            target_features=[1],
            RemoveMissingTarget="No",
        ).fit(X, y)


def test_pct_ssl_classification_selected_weight_attribute_exists():
    rng = np.random.RandomState(5)
    X = rng.normal(size=(25, 2))
    y = (X[:, 0] - 0.5 * X[:, 1] > 0).astype(float).reshape(-1, 1)

    est = PCTClassifier(
        random_state=0,
        ssl=True,
        ssl_method="clus_pct",
        ssl_weight=0.6,
        descriptive_features=[0, 1],
        clustering_features=[0, 1, 2],
        target_features=[2],
        RemoveMissingTarget="No",
    ).fit(X, y)

    assert hasattr(est, "ssl_weight_")
    assert est.ssl_weight_ == pytest.approx(0.6)


def test_pct_ssl_classification_supervised_still_rejects_clustering_x():
    rng = np.random.RandomState(6)
    X = rng.normal(size=(20, 2))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)

    with pytest.raises(NotImplementedError, match="Supervised PCT classification v1 does not support clustering_features"):
        PCTClassifier(
            random_state=0,
            ssl=False,
            descriptive_features=[0, 1],
            clustering_features=[0, 1, 2],
            target_features=[2],
            RemoveMissingTarget="No",
        ).fit(X, y)


def test_pct_ssl_classification_supervised_requires_target_y_equals_clustering_y():
    rng = np.random.RandomState(7)
    X = rng.normal(size=(20, 2))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)

    with pytest.raises(NotImplementedError, match="Supervised PCT classification v1 requires target_y == clustering_y"):
        PCTClassifier(
            random_state=0,
            ssl=False,
            descriptive_features=[0, 1],
            clustering_features=[],
            target_features=[2],
            RemoveMissingTarget="No",
        ).fit(X, y)