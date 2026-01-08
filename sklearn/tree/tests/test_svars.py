import numpy as np
import pytest

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import PCTRegressor


def test_svars_smoke_single_output():
    X = np.array(
        [
            [1.0, 10.0, 0.0, 5.0],
            [2.0, 20.0, 1.0, 6.0],
            [3.0, 30.0, 0.0, 7.0],
            [4.0, 40.0, 1.0, 8.0],
        ],
        dtype=float,
    )
    y = np.array([1.0, 2.0, 2.5, 4.0], dtype=float)

    reg = DecisionTreeRegressor(criterion="svars", random_state=0, max_depth=2)
    reg.fit(X, y)

    pred = reg.predict(X[:2])
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


def test_svars_smoke_multioutput():
    X = np.array(
        [
            [1.0, 10.0, 0.0, 5.0],
            [2.0, 20.0, 1.0, 6.0],
            [3.0, 30.0, 0.0, 7.0],
            [4.0, 40.0, 1.0, 8.0],
            [5.0, 50.0, 0.0, 9.0],
        ],
        dtype=float,
    )
    y = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [2.5, 30.0],
            [4.0, 40.0],
            [4.5, 50.0],
        ],
        dtype=float,
    )

    reg = DecisionTreeRegressor(criterion="svars", random_state=0, max_depth=3)
    reg.fit(X, y)

    pred = reg.predict(X[:2])
    assert pred.shape == (2, 2)
    assert np.isfinite(pred).all()

def test_svars_ftest_default_1_disables_gating():
    rng = np.random.RandomState(0)
    n = 80
    X = rng.normal(size=(n, 4))
    y = 0.1 * X[:, 0] + rng.normal(scale=1.0, size=n)

    reg = PCTRegressor(criterion="svars", random_state=0, max_depth=3, ftest=1.0)
    reg.fit(X, y)
    assert reg.tree_.node_count > 1


def test_svars_ftest_strict_reduces_splitting():
    rng = np.random.RandomState(0)
    n = 80
    X = rng.normal(size=(n, 4))
    y = 0.01 * X[:, 0] + rng.normal(scale=1.0, size=n)

    reg0 = PCTRegressor(criterion="svars", random_state=0, max_depth=3, ftest=1.0)
    reg0.fit(X, y)
    n_nodes0 = reg0.tree_.node_count

    reg1 = PCTRegressor(criterion="svars", random_state=0, max_depth=3, ftest=0.001)
    reg1.fit(X, y)
    n_nodes1 = reg1.tree_.node_count

    assert n_nodes1 <= n_nodes0



def test_svars_impurity_matches_known_values_setting_a():
    X = np.array([
        [1.0, 10.0, 0.0, 5.0],
        [2.0, 20.0, 1.0, 6.0],
        [3.0, 30.0, 0.0, 7.0],
        [4.0, 40.0, 1.0, 8.0],
        [5.0, 50.0, 0.0, 9.0],
        [6.0, 60.0, 1.0, 10.0],
    ], dtype=float)

    Y = np.array([
        [11.0,  1.5],
        [13.0,  2.5],
        [15.0,  3.5],
        [31.0,  8.5],
        [33.0,  9.5],
        [35.0, 10.5],
    ], dtype=float)

    reg = PCTRegressor(
        criterion="svars",
        splitter="best",
        max_depth=3,
        min_samples_leaf=2,
        random_state=0,
        ftest=1.0,
        split_position="clus_exact",
        tie_break="clus",
        missing_target_attr_handling="default_model",
        missing_clustering_attr_handling="estimate_from_parent_node",
    ).fit(X, Y)

    # Expected SVarS impurities:
    # root: 115.5833333333
    # left leaf: 3.3333333333
    # right leaf: 3.3333333333
    imp = reg.tree_.impurity
    assert reg.tree_.node_count == 3
    assert np.allclose(imp[0], 115.58333333333333, rtol=0, atol=1e-10)
    assert np.allclose(imp[1], 3.3333333333333335, rtol=0, atol=1e-10)
    assert np.allclose(imp[2], 3.3333333333333335, rtol=0, atol=1e-10)