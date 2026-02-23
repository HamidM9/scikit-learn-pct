import numpy as np
import pytest
from sklearn.tree import PCTClassifier


def expected_clus_modified_entropy_sum(y, n_classes_per_output):
    """
    Expected CLUS modified entropy:
      prob = (count + 1) / (total + n_classes)
      entropy = - sum_{count>0} prob * log2(prob)

    Important: CLUS skips terms where count == 0 (even though prob would be nonzero).
    Computed as SUM over outputs (not averaged).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    tot = 0.0
    for o in range(y.shape[1]):
        col = y[:, o]
        col = col[~np.isnan(col)].astype(int)
        if col.size == 0:
            continue

        counts = np.bincount(col, minlength=int(n_classes_per_output[o])).astype(float)
        total = counts.sum()
        denom = total + float(n_classes_per_output[o])

        probs = (counts + 1.0) / denom
        probs = probs[counts > 0]  # CLUS skip-zero behavior
        tot += -(probs * np.log2(probs)).sum()

    return tot


def test_clus_modified_entropy_matches_per_output_totals_and_log2():
    X = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
    y = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    clf = PCTClassifier(
        criterion="clus_modified_entropy",
        max_depth=1,
        random_state=0,
        missing_target_attr_handling="parent_node",
    ).fit(X, y)

    got = float(clf.tree_.impurity[0])
    n_classes = list(clf.n_classes_) if hasattr(clf, "n_classes_") else [2, 2]
    exp = expected_clus_modified_entropy_sum(y, n_classes_per_output=n_classes)

    assert got == pytest.approx(exp, rel=1e-12, abs=1e-12)


def test_clus_modified_entropy_differs_from_clus_entropy_with_missing():
    """
    Diagnostic test:
    With missing targets, clus_modified_entropy must still differ from clus_entropy
    whenever there is at least one labeled target per output, because Laplace smoothing
    changes probabilities.

    If this fails, ClusModifiedEntropy is behaving like ClusEntropy in the missing-target path.
    """
    X = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
    y_missing = np.array(
        [
            [0.0, 0.0],
            [0.0, np.nan],
            [1.0, 1.0],
            [np.nan, 1.0],
        ],
        dtype=float,
    )

    clf_ent = PCTClassifier(
        criterion="clus_entropy",
        max_depth=1,
        random_state=0,
        missing_target_attr_handling="parent_node",
    ).fit(X, y_missing)

    clf_me = PCTClassifier(
        criterion="clus_modified_entropy",
        max_depth=1,
        random_state=0,
        missing_target_attr_handling="parent_node",
    ).fit(X, y_missing)

    assert float(clf_me.tree_.impurity[0]) != float(clf_ent.tree_.impurity[0])