import numpy as np

from sklearn.tree import PCTRegressor


def test_pct_regressor_ftest_stronger_value_reduces_growth():
    rng = np.random.RandomState(0)

    X = rng.normal(size=(80, 2))
    y = rng.normal(size=(80, 2))

    reg_no_gate = PCTRegressor(
        criterion="svars",
        random_state=0,
        ftest=1.0,
        min_samples_leaf=2,
    )

    reg_gate = PCTRegressor(
        criterion="svars",
        random_state=0,
        ftest=0.001,
        min_samples_leaf=2,
    )

    reg_no_gate.fit(X, y)
    reg_gate.fit(X, y)

    assert reg_gate.tree_.node_count <= reg_no_gate.tree_.node_count