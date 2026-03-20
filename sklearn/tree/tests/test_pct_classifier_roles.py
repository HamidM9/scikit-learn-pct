import numpy as np
import pytest
from numpy.testing import assert_array_equal

# Adjust this import if your estimator lives somewhere else.
from sklearn.tree import PCTClassifier


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _combined_y_index(x_n_features, y_col):
    """
    In your current role resolution logic, feature-role indices refer to the
    combined schema [X | y]. So if X has p columns, then y-column j has
    combined index p + j.
    """
    return x_n_features + y_col


def _make_two_feature_two_output_dataset():
    """
    X[:, 0] perfectly predicts y[:, 0]
    X[:, 1] perfectly predicts y[:, 1]

    This dataset is ideal for checking whether clustering_y really drives
    split selection.

    Combined schema indices when X has 2 cols:
        X cols: 0, 1
        y cols: 2, 3
    """
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    y0 = X[:, 0].astype(int)
    y1 = X[:, 1].astype(int)
    Y = np.c_[y0, y1]
    return X, Y


def _make_feature_restriction_dataset():
    """
    X[:, 0] is a perfect predictor of y.
    X[:, 1] is only moderately predictive of y.

    This is useful to verify that if descriptive_features=[1], the tree is
    forced to split on feature 1 even though feature 0 would be better.

    y = [0,0,0,0,1,1,1,1]
    x0 = perfect
    x1 = weaker but still useful
    """
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

    x0 = y.astype(float)  # perfect
    x1 = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=float)  # weaker but useful

    X = np.c_[x0, x1]
    Y = y.reshape(-1, 1)
    return X, Y


def _make_missing_target_dataset():
    """
    Simple binary classification dataset with missing values in y.
    Use float y so NaN is representable.
    """
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    y = np.array([0.0, 0.0, np.nan, 1.0, 0.0, np.nan], dtype=float)
    return X, y.reshape(-1, 1)


def _build_pct_classifier(
    X,
    Y,
    *,
    descriptive_features,
    clustering_features,
    target_features,
    criterion="clus_gini",
    max_depth=1,
    random_state=0,
    missing_target_attr_handling="zero",
):
    return PCTClassifier(
        criterion=criterion,
        splitter="best",
        max_depth=max_depth,
        random_state=random_state,
        descriptive_features=descriptive_features,
        clustering_features=clustering_features,
        target_features=target_features,
        missing_target_attr_handling=missing_target_attr_handling,
    )


# ---------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_single_output_basic_fit_predict(criterion):
    """
    Basic smoke test:
    - fit should run
    - predict should run
    - tree should make the expected root split
    """
    X, Y = _make_feature_restriction_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 0)],
        target_features=[_combined_y_index(p, 0)],
        criterion=criterion,
        max_depth=1,
    )
    clf.fit(X, Y)

    pred = clf.predict(X)
    assert pred.shape[0] == X.shape[0]
    assert clf.tree_.feature[0] in (0, 1)


@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_multioutput_basic_fit_predict(criterion):
    """
    Multi-output smoke test:
    - fit should run
    - predict should return an array with same number of rows
    - tree should split on one of the descriptive X features
    """
    X, Y = _make_two_feature_two_output_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 0), _combined_y_index(p, 1)],
        target_features=[_combined_y_index(p, 0), _combined_y_index(p, 1)],
        criterion=criterion,
        max_depth=1,
    )
    clf.fit(X, Y)

    pred = clf.predict(X)
    assert pred.shape[0] == X.shape[0]
    assert clf.tree_.feature[0] in (0, 1)


# ---------------------------------------------------------------------
# Role-validation tests: unsupported layouts must raise
# ---------------------------------------------------------------------

@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_rejects_descriptive_y(criterion):
    """
    classification v1 must reject descriptive_features that point to y-columns
    """
    X, Y = _make_feature_restriction_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, _combined_y_index(p, 0)],  # invalid
        clustering_features=[_combined_y_index(p, 0)],
        target_features=[_combined_y_index(p, 0)],
        criterion=criterion,
    )

    with pytest.raises(NotImplementedError, match="descriptive_y"):
        clf.fit(X, Y)


@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_rejects_clustering_x(criterion):
    """
    classification v1 must reject clustering_features that point to X-columns
    """
    X, Y = _make_feature_restriction_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[0],  # invalid
        target_features=[_combined_y_index(p, 0)],
        criterion=criterion,
    )

    with pytest.raises(NotImplementedError, match="clustering_x"):
        clf.fit(X, Y)


@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_rejects_target_x(criterion):
    """
    classification v1 must reject target_features that point to X-columns
    """
    X, Y = _make_feature_restriction_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 0)],
        target_features=[1],  # invalid
        criterion=criterion,
    )

    with pytest.raises(NotImplementedError, match="target_x"):
        clf.fit(X, Y)


@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_requires_target_y_equal_clustering_y(criterion):
    """
    classification v1 must enforce target_y == clustering_y
    """
    X, Y = _make_two_feature_two_output_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 0)],
        target_features=[_combined_y_index(p, 1)],  # invalid mismatch
        criterion=criterion,
    )

    with pytest.raises(NotImplementedError, match="target_y == clustering_y"):
        clf.fit(X, Y)


# ---------------------------------------------------------------------
# Splitter behavior: only descriptive_x may be used for splits
# ---------------------------------------------------------------------

@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_root_split_uses_only_allowed_descriptive_feature(criterion):
    """
    Only feature 1 is descriptive, so the tree must split on feature 1,
    even though feature 0 would be better globally.
    """
    X, Y = _make_feature_restriction_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[1],  # only feature 1 allowed
        clustering_features=[_combined_y_index(p, 0)],
        target_features=[_combined_y_index(p, 0)],
        criterion=criterion,
        max_depth=1,
    )
    clf.fit(X, Y)

    assert clf.tree_.feature[0] == 1


@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_root_split_can_use_descriptive_feature_zero(criterion):
    """
    If only feature 0 is descriptive, the root split must be on feature 0.
    """
    X, Y = _make_feature_restriction_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0],
        clustering_features=[_combined_y_index(p, 0)],
        target_features=[_combined_y_index(p, 0)],
        criterion=criterion,
        max_depth=1,
    )
    clf.fit(X, Y)

    assert clf.tree_.feature[0] == 0


# ---------------------------------------------------------------------
# Criterion behavior: clustering_y must determine which split is preferred
# ---------------------------------------------------------------------

@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_clustering_y_selects_feature_for_output_0(criterion):
    """
    y[:, 0] depends on X[:, 0]
    y[:, 1] depends on X[:, 1]

    If clustering_y selects output 0, the preferred root split should be X[:, 0].
    """
    X, Y = _make_two_feature_two_output_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 0)],
        target_features=[_combined_y_index(p, 0)],
        criterion=criterion,
        max_depth=1,
    )
    clf.fit(X, Y)

    assert clf.tree_.feature[0] == 0


@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_clustering_y_selects_feature_for_output_1(criterion):
    """
    y[:, 1] depends on X[:, 1]

    If clustering_y selects output 1, the preferred root split should be X[:, 1].
    """
    X, Y = _make_two_feature_two_output_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 1)],
        target_features=[_combined_y_index(p, 1)],
        criterion=criterion,
        max_depth=1,
    )
    clf.fit(X, Y)

    assert clf.tree_.feature[0] == 1


@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_clustering_y_both_outputs_allows_either_relevant_root(criterion):
    """
    If both outputs are in clustering_y, both feature 0 and feature 1 are relevant.
    Depending on tie behavior / scan order / implementation details, either root
    can be acceptable.
    """
    X, Y = _make_two_feature_two_output_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 0), _combined_y_index(p, 1)],
        target_features=[_combined_y_index(p, 0), _combined_y_index(p, 1)],
        criterion=criterion,
        max_depth=1,
    )
    clf.fit(X, Y)

    assert clf.tree_.feature[0] in (0, 1)


# ---------------------------------------------------------------------
# Fitted metadata / role bookkeeping
# ---------------------------------------------------------------------

@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_stores_resolved_roles_before_training(criterion):
    """
    After fitting, the resolved roles and split roles should exist and match
    the expected X/Y partition.
    """
    X, Y = _make_two_feature_two_output_dataset()
    p = X.shape[1]

    descriptive = [0, 1]
    clustering = [_combined_y_index(p, 0), _combined_y_index(p, 1)]
    target = [_combined_y_index(p, 0), _combined_y_index(p, 1)]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=descriptive,
        clustering_features=clustering,
        target_features=target,
        criterion=criterion,
        max_depth=1,
    )
    clf.fit(X, Y)

    assert hasattr(clf, "_pct_feature_roles")
    assert hasattr(clf, "_pct_feature_roles_xy")

    roles_xy = clf._pct_feature_roles_xy

    assert_array_equal(roles_xy["descriptive_x"], np.array([0, 1], dtype=int))
    assert roles_xy["descriptive_y"].size == 0
    assert roles_xy["clustering_x"].size == 0
    assert_array_equal(roles_xy["clustering_y"], np.array([0, 1], dtype=int))
    assert roles_xy["target_x"].size == 0
    assert_array_equal(roles_xy["target_y"], np.array([0, 1], dtype=int))


# ---------------------------------------------------------------------
# Missing-target handling
# ---------------------------------------------------------------------

@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_missing_targets_error_mode_raises(criterion):
    """
    If missing_target_attr_handling='error', any NaN in y should raise.
    """
    X, Y = _make_missing_target_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 0)],
        target_features=[_combined_y_index(p, 0)],
        criterion=criterion,
        missing_target_attr_handling="error",
        max_depth=1,
    )

    with pytest.raises(ValueError, match="Missing targets"):
        clf.fit(X, Y)


@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
@pytest.mark.parametrize("mode", ["zero", "parent_node"])
def test_pct_classifier_missing_targets_supported_modes_fit(criterion, mode):
    """
    Supported non-error missing-target modes should allow fitting to proceed.
    We only check that fit/predict run and that shape is sensible.
    """
    X, Y = _make_missing_target_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 0)],
        target_features=[_combined_y_index(p, 0)],
        criterion=criterion,
        missing_target_attr_handling=mode,
        max_depth=1,
    )
    clf.fit(X, Y)

    pred = clf.predict(X)
    assert pred.shape[0] == X.shape[0]


@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_stores_missing_target_metadata(criterion):
    """
    Your implementation stores:
    - _pct_default_model_
    - _pct_missing_mask_
    - _pct_node_has_obs_
    - _pct_parent_

    This test checks those are created after fit in a missing-target scenario.
    """
    X, Y = _make_missing_target_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 0)],
        target_features=[_combined_y_index(p, 0)],
        criterion=criterion,
        missing_target_attr_handling="zero",
        max_depth=1,
    )
    clf.fit(X, Y)

    assert hasattr(clf, "_pct_default_model_")
    assert hasattr(clf, "_pct_missing_mask_")
    assert hasattr(clf, "_pct_node_has_obs_")
    assert hasattr(clf, "_pct_parent_")

    assert clf._pct_missing_mask_.shape == Y.shape
    assert clf._pct_node_has_obs_.shape[0] == clf.tree_.node_count
    assert clf._pct_parent_.shape[0] == clf.tree_.node_count


# ---------------------------------------------------------------------
# Prediction shape sanity
# ---------------------------------------------------------------------

@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_predict_output_shape_single_output(criterion):
    """
    Single-output shape sanity.
    """
    X, Y = _make_feature_restriction_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 0)],
        target_features=[_combined_y_index(p, 0)],
        criterion=criterion,
        max_depth=1,
    )
    clf.fit(X, Y)
    pred = clf.predict(X)

    assert pred.shape[0] == X.shape[0]


@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_predict_output_shape_multioutput(criterion):
    """
    Multi-output shape sanity.
    """
    X, Y = _make_two_feature_two_output_dataset()
    p = X.shape[1]

    clf = _build_pct_classifier(
        X,
        Y,
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 0), _combined_y_index(p, 1)],
        target_features=[_combined_y_index(p, 0), _combined_y_index(p, 1)],
        criterion=criterion,
        max_depth=1,
    )
    clf.fit(X, Y)
    pred = clf.predict(X)

    assert pred.shape[0] == X.shape[0]


# ---------------------------------------------------------------------
# Deterministic role-driven behavior checks
# ---------------------------------------------------------------------

@pytest.mark.parametrize("criterion", ["clus_gini", "clus_entropy"])
def test_pct_classifier_same_roles_same_root_feature_across_repeated_fits(criterion):
    """
    With a fixed random_state and same data, the chosen root feature should be
    stable across repeated fits.
    """
    X, Y = _make_two_feature_two_output_dataset()
    p = X.shape[1]

    kwargs = dict(
        descriptive_features=[0, 1],
        clustering_features=[_combined_y_index(p, 1)],
        target_features=[_combined_y_index(p, 1)],
        criterion=criterion,
        max_depth=1,
        random_state=0,
    )

    clf1 = _build_pct_classifier(X, Y, **kwargs)
    clf2 = _build_pct_classifier(X, Y, **kwargs)

    clf1.fit(X, Y)
    clf2.fit(X, Y)

    assert clf1.tree_.feature[0] == clf2.tree_.feature[0] == 1