import numpy as np
from sklearn.tree import PCTRegressor

def test_pct_regressor_target_weights_affect_splits():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 3)

    # Two outputs: one strongly linked to X0, another to X1
    y1 = 10.0 * X[:, 0] + rng.randn(400) * 0.1
    y2 = 0.1 * X[:, 1] + rng.randn(400) * 0.1
    Y = np.c_[y1, y2]

    # Unweighted
    a = PCTRegressor(criterion="svars", max_depth=2, random_state=0)
    a.fit(X, Y)
    pred_a = a.predict(X)

    # Weight output 2 heavily -> should bias splits towards X1 signal
    b = PCTRegressor(criterion="svars", target_weights=[0.1, 10.0], max_depth=2, random_state=0)
    b.fit(X, Y)
    pred_b = b.predict(X)

    # Not asserting exact structure, but predictions should differ
    assert not np.allclose(pred_a, pred_b)
