import numpy as np
from sklearn.tree import PCTRegressor


def _make_data():
    # 4 features, 2 targets, with missing targets in different regions
    X = np.array(
        [
            [1.0, 10.0, 0.0, 5.0],
            [2.0, 20.0, 1.0, 6.0],
            [3.0, 30.0, 0.0, 7.0],
            [4.0, 40.0, 1.0, 8.0],
            [5.0, 50.0, 0.0, 9.0],
            [6.0, 60.0, 1.0, 10.0],
        ],
        dtype=float,
    )

    Y = np.array(
        [
            [11.0,  1.5],
            [13.0,  np.nan],   # y2 missing
            [15.0,  np.nan],   # y2 missing
            [31.0,  8.5],
            [np.nan, 9.5],     # y1 missing
            [np.nan, 10.5],    # y1 missing
        ],
        dtype=float,
    )
    return X, Y


def test_missing_target_zero_policy_runs_and_replaces():
    X, Y = _make_data()
    reg = PCTRegressor(
        criterion="svars",
        random_state=0,
        max_depth=3,
        missing_target_attr_handling="zero",
        ftest=1.0,
    )
    reg.fit(X, Y)
    pred = reg.predict(X)

    assert pred.shape == (X.shape[0], 2)
    assert np.isfinite(pred).all()


def test_missing_target_default_model_policy_runs_and_replaces():
    X, Y = _make_data()
    reg = PCTRegressor(
        criterion="svars",
        random_state=0,
        max_depth=3,
        missing_target_attr_handling="default_model",
        ftest=1.0,
    )
    reg.fit(X, Y)
    pred = reg.predict(X)

    assert pred.shape == (X.shape[0], 2)
    assert np.isfinite(pred).all()


def test_missing_target_parent_node_policy_runs_and_replaces():
    X, Y = _make_data()
    reg = PCTRegressor(
        criterion="svars",
        random_state=0,
        max_depth=3,
        missing_target_attr_handling="parent_node",
        ftest=1.0,
    )
    reg.fit(X, Y)
    pred = reg.predict(X)

    assert pred.shape == (X.shape[0], 2)
    assert np.isfinite(pred).all()
