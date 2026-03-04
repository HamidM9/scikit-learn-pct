import numpy as np
import pytest

from numpy.testing import assert_allclose

# Import your estimator from where it lives in your fork.
# If you exposed it in sklearn.tree, replace accordingly.
from sklearn.tree import PCTClassifier


def _grid_4x4_xy():
    """16 samples on a 4x4 grid for x0,x1 in {0,1,2,3}."""
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    X0, X1 = np.meshgrid(xs, xs, indexing="ij")
    X = np.c_[X0.ravel(), X1.ravel()]
    return X


def test_tie_break_clus_prefers_lower_feature_index_on_ties():
    """
    Construct a symmetric classification problem where x0 and x1 give the same
    best split quality at the root. With tie_break='clus', the chosen root feature
    must be the smallest index (x0 -> feature 0), deterministically.

    This test is intentionally sensitive: it should fail if the splitter still
    overwrites on ties or if feature visitation order is randomized under clus mode.
    """
    X = _grid_4x4_xy()

    # Symmetric label rule:
    # class 0 only in the lower-left 2x2 quadrant; else class 1.
    # This creates a situation where splitting first on x0 or x1 is equally good.
    y = ((X[:, 0] > 1.0) | (X[:, 1] > 1.0)).astype(np.intp)

    # Try multiple seeds to ensure determinism is not an artifact of one permutation.
    for seed in [0, 1, 2, 42]:
        clf = PCTClassifier(
            criterion="clus_entropy",
            compat_mode="clus_v1",
            splitter="best",
            split_position="clus_exact",
            tie_break="clus",
            random_state=seed,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,  # important: consider all features
            ccp_alpha=0.0,
        )
        clf.fit(X, y)

        # Root node feature chosen:
        # In sklearn trees: tree_.feature[0] is the root feature index.
        assert clf.tree_.feature[0] == 0, (
            f"Expected root feature 0 under clus tie-break, got {clf.tree_.feature[0]} "
            f"(seed={seed})."
        )


def test_split_position_clus_exact_uses_observed_boundary_not_midpoint():
    """
    On a simple 1D dataset where the best split is between 1 and 2:
      - midpoint mode should yield threshold 1.5
      - clus_exact mode should yield threshold 1.0 (p_prev)
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.intp)

    clf_mid = PCTClassifier(
        criterion="clus_entropy",
        compat_mode="clus_v1",
        splitter="best",
        split_position="midpoint",
        tie_break="clus",
        random_state=0,
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        ccp_alpha=0.0,
    ).fit(X, y)

    clf_exact = PCTClassifier(
        criterion="clus_entropy",
        compat_mode="clus_v1",
        splitter="best",
        split_position="clus_exact",
        tie_break="clus",
        random_state=0,
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        ccp_alpha=0.0,
    ).fit(X, y)

    thr_mid = float(clf_mid.tree_.threshold[0])
    thr_exact = float(clf_exact.tree_.threshold[0])

    assert_allclose(thr_mid, 1.5, rtol=0, atol=1e-12)
    assert_allclose(thr_exact, 1.0, rtol=0, atol=1e-12)