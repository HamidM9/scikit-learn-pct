import numpy as np
from numpy.testing import assert_array_equal

from sklearn.tree import PCTClassifier


def test_pct_role_defaults_last_column_and_clustering_equals_target():
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [0.5, 0.5, 4.0],
            [0.2, 0.8, 5.0],
        ]
    )
    y = np.array([0, 1, 0, 1])

    clf = PCTClassifier()
    clf.fit(X, y)

    roles = clf._pct_feature_roles
    roles_xy = clf._pct_feature_roles_xy

    # combined schema = [X0, X1, X2, y0]
    assert_array_equal(roles["target_features"], np.array([3]))
    assert_array_equal(roles["clustering_features"], np.array([3]))
    assert_array_equal(roles["descriptive_features"], np.array([0, 1, 2]))

    assert_array_equal(roles_xy["target_y"], np.array([0]))
    assert_array_equal(roles_xy["clustering_y"], np.array([0]))
    assert_array_equal(roles_xy["descriptive_x"], np.array([0, 1, 2]))


def test_pct_explicit_overlapping_roles_are_preserved():
    X = np.array(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ]
    )
    y = np.array([0, 1, 0])

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
    assert_array_equal(roles_xy["clustering_y"], np.array([0]))
    assert_array_equal(roles_xy["target_y"], np.array([0]))

def test_pct_explicit_roles_are_resolved_over_combined_schema():
    X = np.array(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ]
    )
    y = np.array([0, 1, 0])

    # combined schema = [0, 1, 2, 3] where 3 is y0
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
    assert_array_equal(roles_xy["clustering_y"], np.array([0]))
    assert_array_equal(roles_xy["target_y"], np.array([0]))