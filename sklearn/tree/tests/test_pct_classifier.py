import numpy as np
import pytest

from sklearn.tree import PCTClassifier

def _toy_multitarget_with_missing():
    # X splits on feature 0
    X = np.array([
        [0.0], [0.1], [0.2],
        [1.0], [1.1], [1.2],
    ])
    # 2 targets; second target missing in left region
    y = np.array([
        [0, np.nan],
        [0, np.nan],
        [1, np.nan],
        [1, 1],
        [1, 1],
        [0, 0],
    ], dtype=float)
    return X, y

@pytest.mark.parametrize("policy", ["default_model", "parent_node", "zero"])
def test_pct_classifier_missing_target_policies(policy):
    X, y = _toy_multitarget_with_missing()

    clf = PCTClassifier(
        criterion="clus_entropy",
        max_depth=1,
        random_state=0,
        missing_target_attr_handling=policy,
    )
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    pred = clf.predict(X)

    # Multi-output => proba is list
    assert isinstance(proba, list)
    assert len(proba) == 2
    assert pred.shape == (X.shape[0], 2)

    # probabilities should be normalized row-wise
    for k in range(2):
        row_sums = proba[k].sum(axis=1)
        assert np.allclose(row_sums, 1.0)

def test_pct_classifier_error_on_missing_when_configured():
    X, y = _toy_multitarget_with_missing()
    clf = PCTClassifier(missing_target_attr_handling="error")
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_pct_classifier_target_weights_affect_split():
    # Construct data where target 0 prefers split on x<=0.5,
    # target 1 prefers split on x<=1.5. Weighting should move split choice.
    X = np.array([[0.0],[0.2],[0.4],[1.0],[1.2],[1.4],[2.0],[2.2],[2.4]])
    y = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
    ], dtype=float)

    # If we ignore target 0, model should focus on target 1 signal (split later)
    clf_focus_t1 = PCTClassifier(
        criterion="clus_entropy",
        max_depth=1,
        random_state=0,
        target_weights=[0.0, 1.0],
        missing_target_attr_handling="default_model",
    )
    clf_focus_t1.fit(X, y)

    # If we focus target 0, split earlier
    clf_focus_t0 = PCTClassifier(
        criterion="clus_entropy",
        max_depth=1,
        random_state=0,
        target_weights=[1.0, 0.0],
        missing_target_attr_handling="default_model",
    )
    clf_focus_t0.fit(X, y)

    # Different split threshold (sanity). In sklearn tree_, root is node 0.
    thr0 = clf_focus_t0.tree_.threshold[0]
    thr1 = clf_focus_t1.tree_.threshold[0]
    assert thr0 != thr1
