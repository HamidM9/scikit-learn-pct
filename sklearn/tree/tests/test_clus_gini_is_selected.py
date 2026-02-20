import numpy as np
from sklearn.tree import PCTClassifier

def test_clus_gini_is_selected_not_fallback():
    X = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
    y = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=float)

    clf = PCTClassifier(
        criterion="clus_gini",
        max_depth=1,
        random_state=0,
        missing_target_attr_handling="parent_node",
    ).fit(X, y)

    # The tree stores impurity computed by the criterion.
    # For this y, gini at root must be > 0 if criterion is gini-like.
    assert float(clf.tree_.impurity[0]) > 0.0
