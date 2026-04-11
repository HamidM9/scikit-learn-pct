# FILE: sklearn/tree/tests/test_pct_remove_missing_target.py

import numpy as np
import pytest

from sklearn.tree import PCTClassifier, PCTRegressor


# ---------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------

def test_pct_classifier_remove_missing_target_in_get_params():
    clf = PCTClassifier(RemoveMissingTarget="Yes")
    params = clf.get_params()

    assert "RemoveMissingTarget" in params
    assert params["RemoveMissingTarget"] == "Yes"


def test_pct_classifier_remove_missing_target_yes_drops_rows_before_error_policy():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, np.nan, 0.0], dtype=np.float64)

    clf = PCTClassifier(
        criterion="clus_entropy",
        RemoveMissingTarget="Yes",
        missing_target_attr_handling="error",
        random_state=0,
    )
    clf.fit(X, y)

    # One row with missing target should be dropped before the "error" policy checks
    assert clf.tree_.n_node_samples[0] == 3


def test_pct_classifier_remove_missing_target_yes_filters_sample_weight():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, np.nan, 0.0], dtype=np.float64)
    sample_weight = np.array([1.0, 2.0, 100.0, 4.0], dtype=np.float64)

    clf = PCTClassifier(
        criterion="clus_entropy",
        RemoveMissingTarget="Yes",
        missing_target_attr_handling="error",
        random_state=0,
    )
    clf.fit(X, y, sample_weight=sample_weight)

    # Missing row has weight 100.0 and should be removed
    assert clf.tree_.weighted_n_node_samples[0] == pytest.approx(7.0)


def test_pct_classifier_remove_missing_target_no_keeps_rows_and_error_raises():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, np.nan, 0.0], dtype=np.float64)

    clf = PCTClassifier(
        criterion="clus_entropy",
        RemoveMissingTarget="No",
        missing_target_attr_handling="error",
        random_state=0,
    )

    with pytest.raises(ValueError, match="Missing targets found"):
        clf.fit(X, y)


def test_pct_classifier_remove_missing_target_no_default_model_keeps_rows():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, np.nan, 0.0], dtype=np.float64)

    clf = PCTClassifier(
        criterion="clus_entropy",
        RemoveMissingTarget="No",
        missing_target_attr_handling="default_model",
        random_state=0,
    )
    clf.fit(X, y)

    # No row removal in this mode
    assert clf.tree_.n_node_samples[0] == 4


def test_pct_classifier_remove_missing_target_yes_multioutput_drops_rows():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [np.nan, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    clf = PCTClassifier(
        criterion="clus_entropy",
        RemoveMissingTarget="Yes",
        missing_target_attr_handling="error",
        random_state=0,
    )
    clf.fit(X, y)

    assert clf.tree_.n_node_samples[0] == 3


def test_pct_classifier_remove_missing_target_no_multioutput_error_raises():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [np.nan, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    clf = PCTClassifier(
        criterion="clus_entropy",
        RemoveMissingTarget="No",
        missing_target_attr_handling="error",
        random_state=0,
    )

    with pytest.raises(ValueError, match="Missing targets found"):
        clf.fit(X, y)


def test_pct_classifier_remove_missing_target_yes_respects_target_features_subset():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)

    # Missing value only in y[:, 0]
    y = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [np.nan, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Use only second output as target; first output should not trigger row removal
    clf = PCTClassifier(
        criterion="clus_entropy",
        target_features=[2],   # combined schema: X has 1 col, so y[:,1] is index 2
        clustering_features=[2],
        RemoveMissingTarget="Yes",
        missing_target_attr_handling="error",
        random_state=0,
    )
    clf.fit(X, y)

    # Since the chosen target column has no missing values, no row should be dropped
    assert clf.tree_.n_node_samples[0] == 4


def test_pct_classifier_remove_missing_target_invalid_value_raises():
    X = np.array([[0.0], [1.0]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)

    clf = PCTClassifier(
        criterion="clus_entropy",
        RemoveMissingTarget="Maybe",
        missing_target_attr_handling="error",
        random_state=0,
    )

    with pytest.raises(ValueError, match="RemoveMissingTarget"):
        clf.fit(X, y)


# ---------------------------------------------------------------------
# REGRESSION
# ---------------------------------------------------------------------

def test_pct_regressor_remove_missing_target_in_get_params():
    reg = PCTRegressor(RemoveMissingTarget="Yes")
    params = reg.get_params()

    assert "RemoveMissingTarget" in params
    assert params["RemoveMissingTarget"] == "Yes"


def test_pct_regressor_remove_missing_target_yes_drops_rows_before_error_policy():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, np.nan, 2.0], dtype=np.float64)

    reg = PCTRegressor(
        RemoveMissingTarget="Yes",
        missing_target="error",
        random_state=0,
    )
    reg.fit(X, y)

    assert reg.tree_.n_node_samples[0] == 3


def test_pct_regressor_remove_missing_target_yes_filters_sample_weight():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, np.nan, 2.0], dtype=np.float64)
    sample_weight = np.array([1.0, 2.0, 100.0, 4.0], dtype=np.float64)

    reg = PCTRegressor(
        RemoveMissingTarget="Yes",
        missing_target="error",
        random_state=0,
    )
    reg.fit(X, y, sample_weight=sample_weight)

    assert reg.tree_.weighted_n_node_samples[0] == pytest.approx(7.0)


def test_pct_regressor_remove_missing_target_no_keeps_rows_and_error_raises():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, np.nan, 2.0], dtype=np.float64)

    reg = PCTRegressor(
        RemoveMissingTarget="No",
        missing_target="error",
        random_state=0,
    )

    with pytest.raises(ValueError, match="Missing targets found"):
        reg.fit(X, y)


def test_pct_regressor_remove_missing_target_no_ignore_keeps_rows():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, np.nan, 2.0], dtype=np.float64)

    reg = PCTRegressor(
        RemoveMissingTarget="No",
        missing_target="ignore",
        random_state=0,
    )
    reg.fit(X, y)

    assert reg.tree_.n_node_samples[0] == 4


def test_pct_regressor_remove_missing_target_yes_multioutput_drops_rows():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [np.nan, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )

    reg = PCTRegressor(
        RemoveMissingTarget="Yes",
        missing_target="error",
        random_state=0,
    )
    reg.fit(X, y)

    assert reg.tree_.n_node_samples[0] == 3


def test_pct_regressor_remove_missing_target_no_multioutput_error_raises():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [np.nan, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )

    reg = PCTRegressor(
        RemoveMissingTarget="No",
        missing_target="error",
        random_state=0,
    )

    with pytest.raises(ValueError, match="Missing targets found"):
        reg.fit(X, y)


def test_pct_regressor_remove_missing_target_yes_respects_target_features_subset():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)

    # Missing value only in y[:, 0]
    y = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [np.nan, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Use only second output as target; first output should not trigger row removal
    reg = PCTRegressor(
        target_features=[2],   # combined schema: X has 1 col, so y[:,1] is index 2
        clustering_features=[2],
        RemoveMissingTarget="Yes",
        missing_target="error",
        random_state=0,
    )
    reg.fit(X, y)

    assert reg.tree_.n_node_samples[0] == 4


def test_pct_regressor_remove_missing_target_invalid_value_raises():
    X = np.array([[0.0], [1.0]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)

    reg = PCTRegressor(
        RemoveMissingTarget="Maybe",
        missing_target="error",
        random_state=0,
    )

    with pytest.raises(ValueError, match="RemoveMissingTarget"):
        reg.fit(X, y)