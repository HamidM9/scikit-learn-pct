import numpy as np
import pytest

from sklearn.tree import PCTRegressor


def _fit_tree(X, Y, missing_handling):
    reg = PCTRegressor(
        criterion="svars",
        splitter="best",
        random_state=0,
        max_depth=1,
        missing_clustering_attr_handling=missing_handling,
    )
    reg.fit(X, Y)
    return reg


def test_svars_equation_single_output_no_missing():
    # y = [1,2,3] => sum=6, sumsq=14, k=3 => svars=14-36/3=2
    X = np.array([[0.0], [1.0], [2.0]])
    Y = np.array([[1.0], [2.0], [3.0]])

    reg = _fit_tree(X, Y, "ignore")
    tree = reg.tree_

    # root impurity should reflect svars averaged over outputs (1 output)
    # Note: depending on whether you divide by eff outputs only, expected is 2.0
    assert tree.impurity[0] == pytest.approx(2.0, rel=1e-12, abs=1e-12)


def test_missing_ignore_skips_fully_missing_output():
    # 2 outputs: y0 valid, y1 fully missing => effective outputs=1
    X = np.array([[0.0], [1.0], [2.0]])
    Y = np.array([
        [1.0, np.nan],
        [2.0, np.nan],
        [3.0, np.nan],
    ])

    reg = _fit_tree(X, Y, "ignore")
    tree = reg.tree_

    # impurity should equal y0 svars (2.0), not average over 2 outputs
    assert tree.impurity[0] == pytest.approx(2.0, rel=1e-12, abs=1e-12)


def test_missing_parent_estimate_uses_parent_svars_for_child():
    # Construct data so that at root both outputs exist,
    # but in left child output 1 is entirely missing.
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    Y = np.array([
        [0.0, 10.0],
        [0.0, 12.0],
        [10.0, np.nan],
        [12.0, np.nan],
    ])

    reg = _fit_tree(X, Y, "estimate_from_parent_node")
    tree = reg.tree_

    # root is node 0; left child is node 1; right child is node 2 (depends on split)
    # We only assert that child impurity is finite (not inf) and not NaN,
    # because parent-estimate should supply a value for the missing output.
    assert np.isfinite(tree.impurity[1])
    assert np.isfinite(tree.impurity[2])


def test_missing_training_estimate_uses_root_cached_value():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    Y = np.array([
        [0.0, 10.0],
        [0.0, 12.0],
        [10.0, np.nan],
        [12.0, np.nan],
    ])

    reg_train = _fit_tree(X, Y, "estimate_from_training_set")
    tree_train = reg_train.tree_

    # same sanity: children impurities should be finite due to training estimate.
    assert np.isfinite(tree_train.impurity[1])
    assert np.isfinite(tree_train.impurity[2])
