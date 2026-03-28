import numpy as np
import pytest

from sklearn.tree import PCTClassifier


def _entropy_bits(labels):
    labels = np.asarray(labels)
    vals, cnts = np.unique(labels, return_counts=True)
    p = cnts / cnts.sum()
    return -(p * np.log2(p)).sum()


def _gini(labels):
    labels = np.asarray(labels)
    vals, cnts = np.unique(labels, return_counts=True)
    p = cnts / cnts.sum()
    return 1.0 - np.square(p).sum()


@pytest.mark.parametrize(
    "criterion, impurity_fn",
    [
        ("clus_entropy", _entropy_bits),
        ("clus_gini", _gini),
    ],
)
def test_pct_root_impurity_uses_sum_over_clustering_outputs_not_average(
    criterion, impurity_fn
):
    # X has one descriptive column
    X = np.array([[0.0], [1.0], [2.0], [3.0]])

    # Two output columns:
    # y0 -> balanced => entropy=1, gini=0.5
    # y1 -> 1/3 vs 3/4 => entropy=0.811278..., gini=0.375
    y = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 1],
        ]
    )

    clf = PCTClassifier(
        criterion=criterion,
        descriptive_features=[0],   # X0
        clustering_features=[1, 2], # y0, y1
        target_features=[2],        # predict only y1
        max_depth=1,
        random_state=0,
    )
    clf.fit(X, y)

    expected = impurity_fn(y[:, 0]) + impurity_fn(y[:, 1])
    assert clf.tree_.impurity[0] == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "criterion, impurity_fn",
    [
        ("clus_entropy", _entropy_bits),
        ("clus_gini", _gini),
    ],
)
def test_pct_target_weights_scale_clustering_impurity(criterion, impurity_fn):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 1],
        ]
    )

    w0 = 2.0
    w1 = 0.5

    clf = PCTClassifier(
        criterion=criterion,
        descriptive_features=[0],
        clustering_features=[1, 2],
        target_features=[2],
        target_weights=[w0, w1],
        max_depth=1,
        random_state=0,
    )
    clf.fit(X, y)

    expected = w0 * impurity_fn(y[:, 0]) + w1 * impurity_fn(y[:, 1])
    assert clf.tree_.impurity[0] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_pct_entropy_ignores_missing_targets_in_clustering_outputs():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])

    # y1 has one missing value
    y = np.array(
        [
            [0.0, 0.0],
            [0.0, np.nan],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    clf = PCTClassifier(
        criterion="clus_entropy",
        descriptive_features=[0],
        clustering_features=[1, 2],  # y0, y1
        target_features=[2],         # predict y1
        missing_target_attr_handling="default_model",
        max_depth=1,
        random_state=0,
    )
    clf.fit(X, y)

    # y0 uses all 4 samples
    expected_y0 = _entropy_bits([0, 0, 1, 1])

    # y1 should ignore the missing one for impurity
    expected_y1 = _entropy_bits([0, 1, 1])

    expected = expected_y0 + expected_y1
    assert clf.tree_.impurity[0] == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "criterion, impurity_fn",
    [
        ("clus_entropy", _entropy_bits),
        ("clus_gini", _gini),
    ],
)
def test_pct_clustering_x_contributes_to_impurity(criterion, impurity_fn):
    # X0 is categorical-like and is used both as descriptive and clustering_x
    X = np.array([[0.0], [0.0], [1.0], [1.0]])

    # predict only y1, but clustering uses X0 and y1
    y = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 1],
        ]
    )

    clf = PCTClassifier(
        criterion=criterion,
        descriptive_features=[0],   # X0
        clustering_features=[0, 2], # X0 and y1
        target_features=[2],        # predict only y1
        max_depth=1,
        random_state=0,
    )
    clf.fit(X, y)

    expected = impurity_fn(X[:, 0]) + impurity_fn(y[:, 1])
    assert clf.tree_.impurity[0] == pytest.approx(expected, rel=1e-12, abs=1e-12)