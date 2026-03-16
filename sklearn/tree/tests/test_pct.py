import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn.tree import PCTClassifier


def _X_3cols_4rows():
    return np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [0.5, 0.5, 4.0],
            [0.2, 0.8, 5.0],
        ]
    )


def _y_binary_4():
    return np.array([0, 1, 0, 1])


def _y_multioutput_4x2():
    return np.array(
        [
            [0, 1],
            [1, 0],
            [0, 0],
            [1, 1],
        ]
    )


def test_pct_role_defaults_last_column_and_clustering_equals_target():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    clf = PCTClassifier()
    clf.fit(X, y)

    roles = clf._pct_feature_roles
    roles_xy = clf._pct_feature_roles_xy

    # combined schema = [X0, X1, X2, y0]
    assert_array_equal(roles["target_features"], np.array([3]))
    assert_array_equal(roles["clustering_features"], np.array([3]))
    assert_array_equal(roles["descriptive_features"], np.array([0, 1, 2]))

    assert_array_equal(roles_xy["descriptive_x"], np.array([0, 1, 2]))
    assert_array_equal(roles_xy["descriptive_y"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["clustering_x"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["clustering_y"], np.array([0]))
    assert_array_equal(roles_xy["target_x"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["target_y"], np.array([0]))


def test_pct_explicit_overlapping_roles_are_preserved():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    # combined schema = [0, 1, 2, 3] where 3 is y0
    clf = PCTClassifier(
        descriptive_features=[0, 3],
        clustering_features=[3],
        target_features=[3],
    )
    clf.fit(X, y)

    roles = clf._pct_feature_roles
    roles_xy = clf._pct_feature_roles_xy

    assert_array_equal(roles["descriptive_features"], np.array([0, 3]))
    assert_array_equal(roles["clustering_features"], np.array([3]))
    assert_array_equal(roles["target_features"], np.array([3]))

    assert_array_equal(roles_xy["descriptive_x"], np.array([0]))
    assert_array_equal(roles_xy["descriptive_y"], np.array([0]))
    assert_array_equal(roles_xy["clustering_x"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["clustering_y"], np.array([0]))
    assert_array_equal(roles_xy["target_x"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["target_y"], np.array([0]))


def test_pct_explicit_roles_are_resolved_over_combined_schema():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    clf = PCTClassifier(
        descriptive_features=[0, 1],
        clustering_features=[3],
        target_features=[3],
    )
    clf.fit(X, y)

    roles = clf._pct_feature_roles
    roles_xy = clf._pct_feature_roles_xy

    assert_array_equal(roles["descriptive_features"], np.array([0, 1]))
    assert_array_equal(roles["clustering_features"], np.array([3]))
    assert_array_equal(roles["target_features"], np.array([3]))

    assert_array_equal(roles_xy["descriptive_x"], np.array([0, 1]))
    assert_array_equal(roles_xy["descriptive_y"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["clustering_y"], np.array([0]))
    assert_array_equal(roles_xy["target_y"], np.array([0]))


def test_pct_duplicate_indices_within_same_role_are_deduplicated():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    clf = PCTClassifier(
        descriptive_features=[0, 0, 2, 2],
        clustering_features=[3, 3],
        target_features=[3, 3],
    )
    clf.fit(X, y)

    roles = clf._pct_feature_roles

    assert_array_equal(roles["descriptive_features"], np.array([0, 2]))
    assert_array_equal(roles["clustering_features"], np.array([3]))
    assert_array_equal(roles["target_features"], np.array([3]))


def test_pct_empty_role_lists_are_accepted():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    clf = PCTClassifier(
        descriptive_features=[],
        clustering_features=[],
        target_features=[],
    )
    clf.fit(X, y)

    roles = clf._pct_feature_roles
    roles_xy = clf._pct_feature_roles_xy

    assert_array_equal(roles["descriptive_features"], np.array([], dtype=np.intp))
    assert_array_equal(roles["clustering_features"], np.array([], dtype=np.intp))
    assert_array_equal(roles["target_features"], np.array([], dtype=np.intp))

    assert_array_equal(roles_xy["descriptive_x"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["descriptive_y"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["clustering_x"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["clustering_y"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["target_x"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["target_y"], np.array([], dtype=np.intp))


def test_pct_target_only_defaults_clustering_to_target_and_descriptive_to_rest():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    clf = PCTClassifier(target_features=[2])
    clf.fit(X, y)

    roles = clf._pct_feature_roles

    assert_array_equal(roles["target_features"], np.array([2]))
    assert_array_equal(roles["clustering_features"], np.array([2]))
    assert_array_equal(roles["descriptive_features"], np.array([0, 1, 3]))


def test_pct_clustering_only_defaults_target_to_last_column():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    clf = PCTClassifier(clustering_features=[1])
    clf.fit(X, y)

    roles = clf._pct_feature_roles

    # combined schema = [0, 1, 2, 3], default target is 3
    assert_array_equal(roles["target_features"], np.array([3]))
    assert_array_equal(roles["clustering_features"], np.array([1]))
    assert_array_equal(roles["descriptive_features"], np.array([0, 2]))


def test_pct_descriptive_only_defaults_target_and_clustering():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    clf = PCTClassifier(descriptive_features=[0, 3])
    clf.fit(X, y)

    roles = clf._pct_feature_roles

    assert_array_equal(roles["descriptive_features"], np.array([0, 3]))
    assert_array_equal(roles["target_features"], np.array([3]))
    assert_array_equal(roles["clustering_features"], np.array([3]))


def test_pct_multioutput_defaults_use_last_combined_column():
    X = _X_3cols_4rows()
    y = _y_multioutput_4x2()

    clf = PCTClassifier()
    clf.fit(X, y)

    roles = clf._pct_feature_roles
    roles_xy = clf._pct_feature_roles_xy

    # combined schema = [X0, X1, X2, y0, y1]
    assert_array_equal(roles["target_features"], np.array([4]))
    assert_array_equal(roles["clustering_features"], np.array([4]))
    assert_array_equal(roles["descriptive_features"], np.array([0, 1, 2, 3]))

    assert_array_equal(roles_xy["target_x"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["target_y"], np.array([1]))
    assert_array_equal(roles_xy["clustering_y"], np.array([1]))
    assert_array_equal(roles_xy["descriptive_x"], np.array([0, 1, 2]))
    assert_array_equal(roles_xy["descriptive_y"], np.array([0]))


def test_pct_multioutput_explicit_target_and_clustering():
    X = _X_3cols_4rows()
    y = _y_multioutput_4x2()

    clf = PCTClassifier(
        descriptive_features=[0, 1],
        clustering_features=[3, 4],
        target_features=[3, 4],
    )
    clf.fit(X, y)

    roles = clf._pct_feature_roles
    roles_xy = clf._pct_feature_roles_xy

    assert_array_equal(roles["descriptive_features"], np.array([0, 1]))
    assert_array_equal(roles["clustering_features"], np.array([3, 4]))
    assert_array_equal(roles["target_features"], np.array([3, 4]))

    assert_array_equal(roles_xy["descriptive_x"], np.array([0, 1]))
    assert_array_equal(roles_xy["clustering_y"], np.array([0, 1]))
    assert_array_equal(roles_xy["target_y"], np.array([0, 1]))


def test_pct_role_index_out_of_range_raises():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    # combined schema = [0, 1, 2, 3]
    clf = PCTClassifier(target_features=[4])

    with pytest.raises(ValueError, match="target_features"):
        clf.fit(X, y)


def test_pct_negative_role_index_raises():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    clf = PCTClassifier(clustering_features=[-1])

    with pytest.raises(ValueError, match="clustering_features"):
        clf.fit(X, y)


def test_pct_non_1d_role_indices_raise():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    clf = PCTClassifier(descriptive_features=[[0, 1], [2, 3]])

    with pytest.raises(ValueError, match="descriptive_features"):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "param_name,param_value",
    [
        ("descriptive_features", [99]),
        ("clustering_features", [99]),
        ("target_features", [99]),
    ],
)
def test_pct_any_role_out_of_range_raises(param_name, param_value):
    X = _X_3cols_4rows()
    y = _y_binary_4()

    kwargs = {param_name: param_value}
    clf = PCTClassifier(**kwargs)

    with pytest.raises(ValueError, match=param_name):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "param_name,param_value",
    [
        ("descriptive_features", [[0, 1]]),
        ("clustering_features", [[0, 1]]),
        ("target_features", [[0, 1]]),
    ],
)
def test_pct_any_role_non_1d_raises(param_name, param_value):
    X = _X_3cols_4rows()
    y = _y_binary_4()

    kwargs = {param_name: param_value}
    clf = PCTClassifier(**kwargs)

    with pytest.raises(ValueError, match=param_name):
        clf.fit(X, y)


def test_pct_explicit_overlap_between_all_three_roles_is_preserved():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    clf = PCTClassifier(
        descriptive_features=[1, 3],
        clustering_features=[2, 3],
        target_features=[3],
    )
    clf.fit(X, y)

    roles = clf._pct_feature_roles

    assert_array_equal(roles["descriptive_features"], np.array([1, 3]))
    assert_array_equal(roles["clustering_features"], np.array([2, 3]))
    assert_array_equal(roles["target_features"], np.array([3]))


def test_pct_explicit_roles_can_include_only_y_side_columns():
    X = _X_3cols_4rows()
    y = _y_multioutput_4x2()

    # combined schema = [0, 1, 2, 3, 4]
    clf = PCTClassifier(
        descriptive_features=[3, 4],
        clustering_features=[4],
        target_features=[3],
    )
    clf.fit(X, y)

    roles_xy = clf._pct_feature_roles_xy

    assert_array_equal(roles_xy["descriptive_x"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["descriptive_y"], np.array([0, 1]))
    assert_array_equal(roles_xy["clustering_x"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["clustering_y"], np.array([1]))
    assert_array_equal(roles_xy["target_x"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["target_y"], np.array([0]))


def test_pct_explicit_roles_can_include_only_x_side_columns():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    clf = PCTClassifier(
        descriptive_features=[0, 1],
        clustering_features=[2],
        target_features=[1],
    )
    clf.fit(X, y)

    roles_xy = clf._pct_feature_roles_xy

    assert_array_equal(roles_xy["descriptive_x"], np.array([0, 1]))
    assert_array_equal(roles_xy["descriptive_y"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["clustering_x"], np.array([2]))
    assert_array_equal(roles_xy["clustering_y"], np.array([], dtype=np.intp))
    assert_array_equal(roles_xy["target_x"], np.array([1]))
    assert_array_equal(roles_xy["target_y"], np.array([], dtype=np.intp))


def test_pct_default_descriptive_can_be_empty():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    # combined schema = [0, 1, 2, 3]
    # target = [3], clustering = [0, 1, 2, 3]
    # descriptive default should become empty
    clf = PCTClassifier(clustering_features=[0, 1, 2, 3])
    clf.fit(X, y)

    roles = clf._pct_feature_roles

    assert_array_equal(roles["target_features"], np.array([3]))
    assert_array_equal(roles["clustering_features"], np.array([0, 1, 2, 3]))
    assert_array_equal(roles["descriptive_features"], np.array([], dtype=np.intp))


def test_pct_multioutput_target_default_uses_last_y_output_not_first():
    X = _X_3cols_4rows()
    y = _y_multioutput_4x2()

    clf = PCTClassifier()
    clf.fit(X, y)

    roles_xy = clf._pct_feature_roles_xy

    # last combined column is y1
    assert_array_equal(roles_xy["target_y"], np.array([1]))
    assert_array_equal(roles_xy["clustering_y"], np.array([1]))


def test_pct_resolved_role_arrays_use_intp_dtype():
    X = _X_3cols_4rows()
    y = _y_binary_4()

    clf = PCTClassifier(
        descriptive_features=[0, 1],
        clustering_features=[3],
        target_features=[3],
    )
    clf.fit(X, y)

    roles = clf._pct_feature_roles
    roles_xy = clf._pct_feature_roles_xy

    assert roles["descriptive_features"].dtype == np.intp
    assert roles["clustering_features"].dtype == np.intp
    assert roles["target_features"].dtype == np.intp

    assert roles_xy["descriptive_x"].dtype == np.intp
    assert roles_xy["descriptive_y"].dtype == np.intp
    assert roles_xy["clustering_x"].dtype == np.intp
    assert roles_xy["clustering_y"].dtype == np.intp
    assert roles_xy["target_x"].dtype == np.intp
    assert roles_xy["target_y"].dtype == np.intp