import numpy as np

from sklearn.tree import PCTClassifier


def test_hmlc_path_consistency():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])

    # hierarchy order:
    # 0 -> root
    # 1 -> root/A
    # 2 -> root/A/B
    Y = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
    ])

    clf = PCTClassifier(
        criterion="clus_gini",
        max_depth=2,
        random_state=0,
        hmlc=True,
        hierarchy=[
            "root",
            "root/A",
            "root/A/B",
        ],
    )

    clf.fit(X, Y)

    pred = clf.predict(X)

    # B => A
    assert np.all(pred[:, 2] <= pred[:, 1])

    # A => root
    assert np.all(pred[:, 1] <= pred[:, 0])



def test_hmlc_depth_weights_are_computed():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])

    Y = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
    ])

    clf = PCTClassifier(
        criterion="clus_gini",
        max_depth=2,
        random_state=0,
        hmlc=True,
        hmlc_weight=0.75,
        hierarchy=[
            "root",
            "root/A",
            "root/A/B",
        ],
    )

    clf.fit(X, Y)

    np.testing.assert_allclose(
        clf._hmlc_label_weights_,
        np.array([1.0, 0.75, 0.75 ** 2]),
    )



def test_hmlc_depth_weights_affect_root_impurity():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])

    Y = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
    ])

    clf_unweighted = PCTClassifier(
        criterion="clus_gini",
        max_depth=1,
        random_state=0,
        hmlc=True,
        hmlc_weight=1.0,
        hierarchy=[
            "root",
            "root/A",
            "root/A/B",
        ],
    )

    clf_weighted = PCTClassifier(
        criterion="clus_gini",
        max_depth=1,
        random_state=0,
        hmlc=True,
        hmlc_weight=0.75,
        hierarchy=[
            "root",
            "root/A",
            "root/A/B",
        ],
    )

    clf_unweighted.fit(X, Y)
    clf_weighted.fit(X, Y)

    assert clf_weighted.tree_.impurity[0] < clf_unweighted.tree_.impurity[0]


def test_hmlc_predict_hmlc_threshold_output_is_consistent():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])

    Y = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
    ])

    clf = PCTClassifier(
        criterion="clus_gini",
        max_depth=2,
        random_state=0,
        hmlc=True,
        hmlc_threshold=0.5,
        hierarchy=[
            "root",
            "root/A",
            "root/A/B",
        ],
    )

    clf.fit(X, Y)

    pred = clf.predict_hmlc(X)

    assert pred.shape == Y.shape
    assert np.all(pred[:, 2] <= pred[:, 1])
    assert np.all(pred[:, 1] <= pred[:, 0])