import numpy as np
import pytest

from sklearn.tree import PCTRegressor


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _non_leaf_features(tree_):
    feat = tree_.feature
    return feat[feat >= 0]


def _make_basic_regression_data():
    # X has 3 features
    # y has 2 targets
    X = np.array(
        [
            [0.0, 10.0, 100.0],
            [0.0, 11.0, 101.0],
            [0.0, 12.0, 102.0],
            [1.0, 20.0, 200.0],
            [1.0, 21.0, 201.0],
            [1.0, 22.0, 202.0],
        ],
        dtype=float,
    )

    y = np.array(
        [
            [1.0, 5.0],
            [1.1, 5.1],
            [1.2, 5.2],
            [9.0, 2.0],
            [9.1, 2.1],
            [9.2, 2.2],
        ],
        dtype=float,
    )
    return X, y


def _make_missing_target_data():
    X = np.array(
        [
            [0.0, 10.0],
            [0.0, 11.0],
            [1.0, 20.0],
            [1.0, 21.0],
        ],
        dtype=float,
    )

    y = np.array(
        [
            [1.0, np.nan],
            [1.5, 5.0],
            [9.0, 2.0],
            [np.nan, 2.5],
        ],
        dtype=float,
    )
    return X, y


# ---------------------------------------------------------------------
# Role resolution tests
# ---------------------------------------------------------------------
def test_pct_regressor_default_roles_use_all_y_for_target_and_clustering_and_x_for_descriptive():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(random_state=0)
    reg.fit(X, y)

    roles = reg._pct_feature_roles
    roles_xy = reg._pct_feature_roles_xy

    # Combined schema = [X0, X1, X2, y0, y1]
    assert np.array_equal(
        roles["target_features"],
        np.array([3, 4], dtype=np.intp),
    )
    assert np.array_equal(
        roles["clustering_features"],
        np.array([3, 4], dtype=np.intp),
    )
    assert np.array_equal(
        roles["descriptive_features"],
        np.array([0, 1, 2], dtype=np.intp),
    )

    assert np.array_equal(
        roles_xy["descriptive_x"],
        np.array([0, 1, 2], dtype=np.intp),
    )
    assert roles_xy["descriptive_y"].size == 0

    assert roles_xy["clustering_x"].size == 0
    assert np.array_equal(
        roles_xy["clustering_y"],
        np.array([0, 1], dtype=np.intp),
    )

    assert roles_xy["target_x"].size == 0
    assert np.array_equal(
        roles_xy["target_y"],
        np.array([0, 1], dtype=np.intp),
    )
def test_pct_regressor_explicit_role_resolution():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        descriptive_features=[0, 2],
        clustering_features=[1, 3, 4],
        target_features=[3, 4],
        random_state=0,
    )
    reg.fit(X, y)

    roles_xy = reg._pct_feature_roles_xy

    assert np.array_equal(roles_xy["descriptive_x"], np.array([0, 2], dtype=np.intp))
    assert roles_xy["descriptive_y"].size == 0

    assert np.array_equal(roles_xy["clustering_x"], np.array([1], dtype=np.intp))
    assert np.array_equal(roles_xy["clustering_y"], np.array([0, 1], dtype=np.intp))

    assert roles_xy["target_x"].size == 0
    assert np.array_equal(roles_xy["target_y"], np.array([0, 1], dtype=np.intp))



def test_pct_regressor_role_indices_are_unique_and_preserve_order():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        descriptive_features=[2, 0, 2, 0],
        clustering_features=[4, 3, 4],
        target_features=[4, 3, 4],
        random_state=0,
    )
    reg.fit(X, y)

    roles = reg._pct_feature_roles
    assert np.array_equal(roles["descriptive_features"], np.array([2, 0], dtype=np.intp))
    assert np.array_equal(roles["clustering_features"], np.array([4, 3], dtype=np.intp))
    assert np.array_equal(roles["target_features"], np.array([4, 3], dtype=np.intp))

@pytest.mark.parametrize(
    "bad_indices",
    [
        [-1],
        [999],
        [0, 1, 999],
    ],
)
def test_pct_regressor_invalid_role_indices_raise(bad_indices):
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        descriptive_features=bad_indices,
        random_state=0,
    )

    with pytest.raises(ValueError, match="outside the valid combined schema range"):
        reg.fit(X, y)


def test_pct_regressor_non_1d_role_indices_raise():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        descriptive_features=[[0, 1]],
        random_state=0,
    )

    with pytest.raises(ValueError, match="must be a 1-dimensional array-like or None"):
        reg.fit(X, y)


# ---------------------------------------------------------------------
# v1 restrictions
# ---------------------------------------------------------------------
def test_pct_regressor_rejects_descriptive_y_in_v1():
    X, y = _make_basic_regression_data()

    # Combined schema [X0,X1,X2,y0,y1], so 3 means y0
    reg = PCTRegressor(
        descriptive_features=[0, 3],
        target_features=[3],
        clustering_features=[3],
        random_state=0,
    )

    with pytest.raises(NotImplementedError, match="descriptive_features that point to y-columns"):
        reg.fit(X, y)


def test_pct_regressor_rejects_target_x_in_v1():
    X, y = _make_basic_regression_data()

    # target_features includes X1
    reg = PCTRegressor(
        descriptive_features=[0, 1, 2],
        clustering_features=[3],
        target_features=[1, 3],
        random_state=0,
    )

    with pytest.raises(NotImplementedError, match="target_features that point to X-columns"):
        reg.fit(X, y)


def test_pct_regressor_rejects_empty_target_features():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1, 2],
        clustering_features=[3],
        target_features=[],
        random_state=0,
    )

    with pytest.raises(ValueError, match="target_features cannot be empty"):
        reg.fit(X, y)


def test_pct_regressor_rejects_empty_clustering_features():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1, 2],
        clustering_features=[],
        target_features=[3],
        random_state=0,
    )

    with pytest.raises(ValueError, match="clustering_features cannot be empty"):
        reg.fit(X, y)


# ---------------------------------------------------------------------
# Splitting must use descriptive_x only
# ---------------------------------------------------------------------
def test_pct_regressor_splits_only_on_descriptive_x_features():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        criterion="svars",
        descriptive_features=[0],      # only X0 may split
        clustering_features=[3, 4],    # impurity from y0,y1
        target_features=[3, 4],        # predict y0,y1
        random_state=0,
        max_depth=3,
    )
    reg.fit(X, y)

    used = _non_leaf_features(reg.tree_)
    assert used.size > 0
    assert np.all(used == 0)


def test_pct_regressor_can_split_on_subset_of_x_features():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        criterion="svars",
        descriptive_features=[2],      # only X2 may split
        clustering_features=[3, 4],
        target_features=[3, 4],
        random_state=0,
        max_depth=3,
    )
    reg.fit(X, y)

    used = _non_leaf_features(reg.tree_)
    assert used.size > 0
    assert np.all(used == 2)


# ---------------------------------------------------------------------
# Prediction dimension must come from target_features
# ---------------------------------------------------------------------
def test_pct_regressor_predict_shape_matches_target_features_multioutput():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1, 2],
        clustering_features=[3, 4],
        target_features=[3, 4],
        random_state=0,
    )
    reg.fit(X, y)

    pred = reg.predict(X)
    assert pred.shape == (X.shape[0], 2)


def test_pct_regressor_predict_shape_matches_target_features_single_output():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1, 2],
        clustering_features=[3, 4],
        target_features=[3],
        random_state=0,
    )
    reg.fit(X, y)

    pred = reg.predict(X)
    assert pred.shape == (X.shape[0],)


def test_pct_regressor_tree_n_outputs_matches_target_outputs():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1, 2],
        clustering_features=[1, 3, 4],  # 3 clustering outputs total: X1, y0, y1
        target_features=[4],            # predict only y1
        random_state=0,
    )
    reg.fit(X, y)

    assert reg.n_outputs_ == 1
    pred = reg.predict(X)
    assert pred.ndim == 1


# ---------------------------------------------------------------------
# Clustering view tests
# ---------------------------------------------------------------------
def test_pct_regressor_clustering_view_can_mix_x_and_y():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1, 2],
        clustering_features=[1, 3, 4],  # X1 + y0 + y1
        target_features=[3, 4],
        random_state=0,
    )
    reg.fit(X, y)

    roles_xy = reg._pct_feature_roles_xy
    assert np.array_equal(roles_xy["clustering_x"], np.array([1], dtype=np.intp))
    assert np.array_equal(roles_xy["clustering_y"], np.array([0, 1], dtype=np.intp))


def test_pct_regressor_target_weights_length_matches_number_of_clustering_outputs():
    X, y = _make_basic_regression_data()

    # clustering outputs = X1 + y0 + y1 => 3 outputs
    reg = PCTRegressor(
        descriptive_features=[0, 1, 2],
        clustering_features=[1, 3, 4],
        target_features=[3, 4],
        target_weights=[1.0, 2.0],  # wrong length
        random_state=0,
    )

    with pytest.raises(ValueError, match="target_weights length .* does not match"):
        reg.fit(X, y)


def test_pct_regressor_target_weights_accept_correct_clustering_length():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1, 2],
        clustering_features=[1, 3, 4],  # 3 clustering outputs
        target_features=[3, 4],
        target_weights=[1.0, 2.0, 3.0],
        random_state=0,
    )
    reg.fit(X, y)

    pred = reg.predict(X)
    assert pred.shape == (X.shape[0], 2)


# ---------------------------------------------------------------------
# Missing target handling in fit
# ---------------------------------------------------------------------
def test_pct_regressor_missing_target_error_raises():
    X, y = _make_missing_target_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1],
        clustering_features=[2, 3],  # y0,y1
        target_features=[2, 3],      # y0,y1
        missing_target="error",
        random_state=0,
    )

    with pytest.raises(ValueError, match="Missing targets found but missing_target='error'"):
        reg.fit(X, y)


def test_pct_regressor_missing_target_ignore_fits():
    X, y = _make_missing_target_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1],
        clustering_features=[2, 3],
        target_features=[2, 3],
        missing_target="ignore",
        random_state=0,
    )
    reg.fit(X, y)

    pred = reg.predict(X)
    assert pred.shape == (X.shape[0], 2)


def test_pct_regressor_default_model_is_mean_of_observed_targets():
    X, y = _make_missing_target_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1],
        clustering_features=[2, 3],
        target_features=[2, 3],
        missing_target="ignore",
        random_state=0,
    )
    reg.fit(X, y)

    # y0 observed: 1.0, 1.5, 9.0 => mean = 11.5 / 3
    # y1 observed: 5.0, 2.0, 2.5 => mean = 9.5 / 3
    expected = np.array([11.5 / 3.0, 9.5 / 3.0])
    assert np.allclose(reg._pct_default_model_, expected)


def test_pct_regressor_node_has_observed_metadata_exists():
    X, y = _make_missing_target_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1],
        clustering_features=[2, 3],
        target_features=[2, 3],
        missing_target="ignore",
        random_state=0,
    )
    reg.fit(X, y)

    assert hasattr(reg, "_pct_node_has_obs_")
    assert reg._pct_node_has_obs_.shape[0] == reg.tree_.node_count
    assert reg._pct_node_has_obs_.shape[1] == 2


def test_pct_regressor_parent_vector_exists():
    X, y = _make_missing_target_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1],
        clustering_features=[2, 3],
        target_features=[2, 3],
        missing_target="ignore",
        random_state=0,
    )
    reg.fit(X, y)

    assert hasattr(reg, "_pct_parent_")
    assert reg._pct_parent_.shape == (reg.tree_.node_count,)
    assert reg._pct_parent_[0] == -1


# ---------------------------------------------------------------------
# Missing target prediction fallback policies
# ---------------------------------------------------------------------
def test_pct_regressor_predict_default_model_policy_returns_finite_values():
    X, y = _make_missing_target_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1],
        clustering_features=[2, 3],
        target_features=[2, 3],
        missing_target="ignore",
        missing_target_attr_handling="default_model",
        random_state=0,
        max_depth=2,
    )
    reg.fit(X, y)

    pred = reg.predict(X)
    assert pred.shape == (X.shape[0], 2)
    assert np.isfinite(pred).all()


def test_pct_regressor_predict_zero_policy_returns_finite_values():
    X, y = _make_missing_target_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1],
        clustering_features=[2, 3],
        target_features=[2, 3],
        missing_target="ignore",
        missing_target_attr_handling="zero",
        random_state=0,
        max_depth=2,
    )
    reg.fit(X, y)

    pred = reg.predict(X)
    assert pred.shape == (X.shape[0], 2)
    assert np.isfinite(pred).all()


def test_pct_regressor_predict_parent_node_policy_returns_finite_values():
    X, y = _make_missing_target_data()

    reg = PCTRegressor(
        descriptive_features=[0, 1],
        clustering_features=[2, 3],
        target_features=[2, 3],
        missing_target="ignore",
        missing_target_attr_handling="parent_node",
        random_state=0,
        max_depth=2,
    )
    reg.fit(X, y)

    pred = reg.predict(X)
    assert pred.shape == (X.shape[0], 2)
    assert np.isfinite(pred).all()


# ---------------------------------------------------------------------
# Exact target-view prediction checks
# ---------------------------------------------------------------------
def test_pct_regressor_single_target_prediction_uses_only_selected_target_column():
    X = np.array(
        [[0.0], [0.0], [1.0], [1.0]],
        dtype=float,
    )
    y = np.array(
        [
            [1.0, 100.0],
            [1.0, 100.0],
            [9.0, -50.0],
            [9.0, -50.0],
        ],
        dtype=float,
    )

    # Predict only y0
    reg0 = PCTRegressor(
        descriptive_features=[0],
        clustering_features=[1, 2],   # y0,y1
        target_features=[1],          # y0 only
        random_state=0,
        max_depth=1,
    )
    reg0.fit(X, y)
    pred0 = reg0.predict(X)
    assert pred0.shape == (4,)
    assert np.allclose(pred0[:2], 1.0)
    assert np.allclose(pred0[2:], 9.0)

    # Predict only y1
    reg1 = PCTRegressor(
        descriptive_features=[0],
        clustering_features=[1, 2],   # y0,y1
        target_features=[2],          # y1 only
        random_state=0,
        max_depth=1,
    )
    reg1.fit(X, y)
    pred1 = reg1.predict(X)
    assert pred1.shape == (4,)
    assert np.allclose(pred1[:2], 100.0)
    assert np.allclose(pred1[2:], -50.0)


def test_pct_regressor_multi_target_prediction_uses_selected_target_columns_in_order():
    X = np.array(
        [[0.0], [0.0], [1.0], [1.0]],
        dtype=float,
    )
    y = np.array(
        [
            [1.0, 100.0],
            [1.0, 100.0],
            [9.0, -50.0],
            [9.0, -50.0],
        ],
        dtype=float,
    )

    reg = PCTRegressor(
        descriptive_features=[0],
        clustering_features=[1, 2],
        target_features=[2, 1],  # predict [y1, y0]
        random_state=0,
        max_depth=1,
    )
    reg.fit(X, y)
    pred = reg.predict(X)

    assert pred.shape == (4, 2)
    assert np.allclose(pred[:2, 0], 100.0)
    assert np.allclose(pred[:2, 1], 1.0)
    assert np.allclose(pred[2:, 0], -50.0)
    assert np.allclose(pred[2:, 1], 9.0)


# ---------------------------------------------------------------------
# Misc smoke tests
# ---------------------------------------------------------------------
def test_pct_regressor_accepts_single_output_y_vector():
    X = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=float)
    y = np.array([1.0, 1.0, 9.0, 9.0], dtype=float)

    # Combined schema = [X0 | y0], so target_features=[1]
    reg = PCTRegressor(
        descriptive_features=[0],
        clustering_features=[1],
        target_features=[1],
        random_state=0,
        max_depth=1,
    )
    reg.fit(X, y)

    pred = reg.predict(X)
    assert pred.shape == (4,)
    assert np.allclose(pred[:2], 1.0)
    assert np.allclose(pred[2:], 9.0)


def test_pct_regressor_can_use_clustering_x_only_and_target_y_only():
    X = np.array(
        [[0.0, 10.0], [0.0, 11.0], [1.0, 20.0], [1.0, 21.0]],
        dtype=float,
    )
    y = np.array([1.0, 1.0, 9.0, 9.0], dtype=float)

    # Combined schema = [X0, X1 | y0]
    reg = PCTRegressor(
        descriptive_features=[0],
        clustering_features=[1],  # only X1 drives impurity
        target_features=[2],      # predict y0
        random_state=0,
        max_depth=1,
    )
    reg.fit(X, y)

    pred = reg.predict(X)
    assert pred.shape == (4,)
    assert np.isfinite(pred).all()


def test_pct_regressor_can_use_clustering_y_only_and_target_y_only():
    X = np.array(
        [[0.0], [0.0], [1.0], [1.0]],
        dtype=float,
    )
    y = np.array(
        [
            [1.0, 10.0],
            [1.0, 10.0],
            [9.0, 20.0],
            [9.0, 20.0],
        ],
        dtype=float,
    )

    reg = PCTRegressor(
        descriptive_features=[0],
        clustering_features=[1, 2],  # y0, y1
        target_features=[1],         # predict y0
        random_state=0,
        max_depth=1,
    )
    reg.fit(X, y)

    pred = reg.predict(X)
    assert pred.shape == (4,)
    assert np.allclose(pred[:2], 1.0)
    assert np.allclose(pred[2:], 9.0)


def test_pct_regressor_stored_descriptive_roles_are_not_mutated_by_fit():
    X, y = _make_basic_regression_data()

    reg = PCTRegressor(random_state=0)
    reg.fit(X, y)

    assert np.array_equal(
        reg._pct_feature_roles_xy["descriptive_x"],
        np.array([0, 1, 2], dtype=np.intp),
    )