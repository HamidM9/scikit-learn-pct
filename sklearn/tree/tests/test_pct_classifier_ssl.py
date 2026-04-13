import numpy as np
import pytest

from sklearn.tree import PCTClassifier


def _tiny_ssl_classification_data():
    # First 3 columns are X, last 3 columns are Y
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


def test_pct_classifier_ssl_param_in_get_params():
    clf = PCTClassifier(ssl=True)
    params = clf.get_params()

    assert "ssl" in params
    assert params["ssl"] is True


def test_pct_classifier_remove_missing_target_yes_drops_rows():
    X, y = _tiny_ssl_classification_data()

    clf = PCTClassifier(
        ssl=False,
        RemoveMissingTarget="Yes",
        missing_target_attr_handling="error",
        random_state=0,
    )
    clf.fit(X, y)

    # Only fully observed rows remain: 0,1,2,3
    assert clf.tree_.n_node_samples[0] == 4


def test_pct_classifier_remove_missing_target_no_ssl_false_error_raises():
    X, y = _tiny_ssl_classification_data()

    clf = PCTClassifier(
        ssl=False,
        RemoveMissingTarget="No",
        missing_target_attr_handling="error",
        random_state=0,
    )

    with pytest.raises(ValueError, match="ssl=False"):
        clf.fit(X, y)


def test_pct_classifier_remove_missing_target_no_ssl_true_allows_missing_targets():
    X, y = _tiny_ssl_classification_data()

    clf = PCTClassifier(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target_attr_handling="error",
        random_state=0,
    )
    clf.fit(X, y)

    assert clf.tree_.n_node_samples[0] == 8


def test_pct_classifier_ssl_sets_missing_masks():
    X, y = _tiny_ssl_classification_data()

    clf = PCTClassifier(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target_attr_handling="error",
        random_state=0,
    )
    clf.fit(X, y)

    assert hasattr(clf, "_pct_missing_mask_")
    assert hasattr(clf, "_pct_missing_mask_clust_")
    assert clf._pct_missing_mask_.shape == (8, 3)
    assert clf._pct_missing_mask_clust_.shape[0] == 8


def test_pct_classifier_ssl_sets_default_model_majority():
    X, y = _tiny_ssl_classification_data()

    clf = PCTClassifier(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target_attr_handling="error",
        random_state=0,
    )
    clf.fit(X, y)

    assert hasattr(clf, "_pct_default_model_")
    assert clf._pct_default_model_.shape == (3,)

    # col0 observed = [0,0,1,0] -> majority 0
    # col1 observed = [1,1,1,0,1,1] -> majority 1
    # col2 observed = [1,1,0,0,1] -> majority 1
    assert np.array_equal(clf._pct_default_model_, np.array([0, 1, 1], dtype=np.intp))


def test_pct_classifier_ssl_predict_proba_runs():
    X, y = _tiny_ssl_classification_data()

    clf = PCTClassifier(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target_attr_handling="default_model",
        random_state=0,
    )
    clf.fit(X, y)

    proba = clf.predict_proba(X)

    assert isinstance(proba, list)
    assert len(proba) == 3
    assert proba[0].shape[0] == 8
    assert np.isfinite(proba[0]).all()


def test_pct_classifier_ssl_predict_runs():
    X, y = _tiny_ssl_classification_data()

    clf = PCTClassifier(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target_attr_handling="default_model",
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X)

    assert pred.shape == (8, 3)


def test_pct_classifier_ssl_parent_node_predict_runs():
    X, y = _tiny_ssl_classification_data()

    clf = PCTClassifier(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target_attr_handling="parent_node",
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X)
    assert pred.shape == (8, 3)


def test_pct_classifier_ssl_zero_policy_predict_runs():
    X, y = _tiny_ssl_classification_data()

    clf = PCTClassifier(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target_attr_handling="zero",
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X)
    assert pred.shape == (8, 3)


def test_pct_classifier_remove_missing_target_invalid_value_raises():
    X, y = _tiny_ssl_classification_data()

    clf = PCTClassifier(
        ssl=False,
        RemoveMissingTarget="Maybe",
        missing_target_attr_handling="error",
        random_state=0,
    )

    with pytest.raises(ValueError, match="RemoveMissingTarget"):
        clf.fit(X, y)


def test_pct_classifier_ssl_respects_target_features_subset():
    X, y = _tiny_ssl_classification_data()

    # Combined schema [X0, X1, X2, Y0, Y1, Y2]
    # Use only Y1 and Y2 as target/clustering
    clf = PCTClassifier(
        ssl=True,
        RemoveMissingTarget="No",
        missing_target_attr_handling="error",
        target_features=[4, 5],
        clustering_features=[4, 5],
        random_state=0,
    )
    clf.fit(X, y)

    assert clf.tree_.n_node_samples[0] == 8
    assert clf._pct_missing_mask_.shape == (8, 2)


def test_pct_classifier_remove_missing_target_yes_respects_target_features_subset():
    X, y = _tiny_ssl_classification_data()

    clf = PCTClassifier(
        ssl=False,
        RemoveMissingTarget="Yes",
        missing_target_attr_handling="error",
        target_features=[4, 5],
        clustering_features=[4, 5],
        random_state=0,
    )
    clf.fit(X, y)

    # Rows kept: 0,1,2,3,4
    assert clf.tree_.n_node_samples[0] == 5