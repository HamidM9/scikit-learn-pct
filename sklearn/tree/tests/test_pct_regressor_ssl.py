# FILE: sklearn/tree/tests/test_pct_regressor_ssl.py

import numpy as np
import pytest

from sklearn.tree import PCTRegressor


def _tiny_ssl_regression_data():
    # First 3 columns are X, last 3 are Y
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    y = np.array(
        [
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [np.nan, 1.0, 1.0],
            [np.nan, 1.0, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    return X, y


def test_pct_regressor_ssl_param_in_get_params():
    reg = PCTRegressor(ssl=True)
    params = reg.get_params()

    assert "ssl" in params
    assert params["ssl"] is True


def test_pct_regressor_remove_missing_target_yes_drops_rows():
    X, y = _tiny_ssl_regression_data()

    reg = PCTRegressor(
        ssl=False,
        RemoveMissingTarget="Yes",
        missing_target="error",
        random_state=0,
    )
    reg.fit(X, y)

    # Keep only rows with fully observed targets:
    # rows 0,1,2,3
    assert reg.tree_.n_node_samples[0] == 4


def test_pct_regressor_remove_missing_target_yes_filters_sample_weight():
    X, y = _tiny_ssl_regression_data()
    sample_weight = np.array([1, 1, 1, 1, 10, 10, 10, 10], dtype=np.float64)

    reg = PCTRegressor(
        ssl=False,
        RemoveMissingTarget="Yes",
        missing_target="error",
        random_state=0,
    )
    reg.fit(X, y, sample_weight=sample_weight)

    # Only first 4 rows remain
    assert reg.tree_.weighted_n_node_samples[0] == pytest.approx(4.0)


def test_pct_regressor_remove_missing_target_no_ssl_false_error_raises():
    X, y = _tiny_ssl_regression_data()

    reg = PCTRegressor(
        ssl=False,
        RemoveMissingTarget="No",
        missing_target="error",
        random_state=0,
    )

    with pytest.raises(ValueError, match="ssl=False"):
        reg.fit(X, y)


def test_pct_regressor_remove_missing_target_no_ssl_true_allows_missing_targets():
    X, y = _tiny_ssl_regression_data()

    reg = PCTRegressor(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target="error",
        random_state=0,
    )
    reg.fit(X, y)

    # SSL path should keep all rows
    assert reg.tree_.n_node_samples[0] == 8


def test_pct_regressor_remove_missing_target_no_missing_target_ignore_keeps_rows():
    X, y = _tiny_ssl_regression_data()

    reg = PCTRegressor(
        ssl=False,
        RemoveMissingTarget="No",
        missing_target="ignore",
        random_state=0,
    )
    reg.fit(X, y)

    assert reg.tree_.n_node_samples[0] == 8


def test_pct_regressor_ssl_sets_missing_masks():
    X, y = _tiny_ssl_regression_data()

    reg = PCTRegressor(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target="error",
        random_state=0,
    )
    reg.fit(X, y)

    assert hasattr(reg, "_pct_missing_mask_")
    assert hasattr(reg, "_pct_missing_mask_clust_")

    assert reg._pct_missing_mask_.shape == (8, 3)
    assert reg._pct_missing_mask_clust_.shape[0] == 8


def test_pct_regressor_ssl_sets_default_model():
    X, y = _tiny_ssl_regression_data()

    reg = PCTRegressor(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target="error",
        random_state=0,
    )
    reg.fit(X, y)

    assert hasattr(reg, "_pct_default_model_")
    assert reg._pct_default_model_.shape == (3,)

    # From observed rows:
    # col0 observed = [0,0,1,0] mean = 0.25
    # col1 observed = [1,1,1,0,1,1] mean = 5/6
    # col2 observed = [1,1,0,0,1] mean = 3/5
    assert reg._pct_default_model_[0] == pytest.approx(0.25)
    assert reg._pct_default_model_[1] == pytest.approx(5.0 / 6.0)
    assert reg._pct_default_model_[2] == pytest.approx(3.0 / 5.0)


def test_pct_regressor_ssl_predict_runs_on_training_data():
    X, y = _tiny_ssl_regression_data()

    reg = PCTRegressor(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target="error",
        missing_target_attr_handling="default_model",
        random_state=0,
    )
    reg.fit(X, y)

    pred = reg.predict(X)

    assert pred.shape == (8, 3)
    assert np.isfinite(pred).all()


def test_pct_regressor_ssl_parent_node_predict_runs():
    X, y = _tiny_ssl_regression_data()

    reg = PCTRegressor(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target="error",
        missing_target_attr_handling="parent_node",
        random_state=0,
    )
    reg.fit(X, y)

    pred = reg.predict(X)

    assert pred.shape == (8, 3)
    assert np.isfinite(pred).all()


def test_pct_regressor_ssl_zero_policy_predict_runs():
    X, y = _tiny_ssl_regression_data()

    reg = PCTRegressor(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target="error",
        missing_target_attr_handling="zero",
        random_state=0,
    )
    reg.fit(X, y)

    pred = reg.predict(X)

    assert pred.shape == (8, 3)
    assert np.isfinite(pred).all()


def test_pct_regressor_remove_missing_target_invalid_value_raises():
    X, y = _tiny_ssl_regression_data()

    reg = PCTRegressor(
        ssl=False,
        RemoveMissingTarget="Maybe",
        missing_target="error",
        random_state=0,
    )

    with pytest.raises(ValueError, match="RemoveMissingTarget"):
        reg.fit(X, y)


def test_pct_regressor_ssl_respects_target_features_subset():
    X, y = _tiny_ssl_regression_data()

    # Combined schema is [X0,X1,X2,Y0,Y1,Y2]
    # Use only Y1 and Y2 as targets/clustering
    reg = PCTRegressor(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target="error",
        target_features=[4, 5],
        clustering_features=[4, 5],
        random_state=0,
    )
    reg.fit(X, y)

    assert reg.tree_.n_node_samples[0] == 8
    assert reg._pct_missing_mask_.shape == (8, 2)


def test_pct_regressor_remove_missing_target_yes_respects_target_features_subset():
    X, y = _tiny_ssl_regression_data()

    # Use only Y1 and Y2 as targets/clustering.
    # Rows 6 and 7 have missing in both Y1/Y2.
    # Row 5 has missing in Y2.
    # Row 4 is fully observed on Y1/Y2.
    reg = PCTRegressor(
        ssl=False,
        RemoveMissingTarget="Yes",
        missing_target="error",
        target_features=[4, 5],
        clustering_features=[4, 5],
        random_state=0,
    )
    reg.fit(X, y)

    # Rows kept: 0,1,2,3,4
    assert reg.tree_.n_node_samples[0] == 5