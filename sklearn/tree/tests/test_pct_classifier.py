import numpy as np
from sklearn.tree import PCTClassifier

def test_pct_classifier_smoke():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)
    y = (X[:, 0] + 0.1 * rng.randn(200) > 0).astype(int)

    clf = PCTClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)

    pred = clf.predict(X[:10])
    assert pred.shape == (10,)
    assert set(np.unique(pred)).issubset({0, 1})
