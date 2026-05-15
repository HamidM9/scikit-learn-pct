import numpy as np

from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error


class ClusXValSelection:
    """Deterministic CLUS-style cross-validation fold selection."""

    def __init__(self, n_splits=10, random_state=0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        self.n_splits = int(n_splits)
        self.random_state = random_state

    def make_folds(self, n_samples):
        """Replicate CLUS XValRandomSelection fold assignment."""
        n_samples = int(n_samples)

        if self.n_splits > n_samples:
            raise ValueError(
                f"n_splits={self.n_splits} cannot be larger than "
                f"n_samples={n_samples}."
            )

        # Leave-one-out behavior from CLUS
        if self.n_splits == n_samples:
            return np.arange(n_samples, dtype=np.intp)

        rng = np.random.RandomState(self.random_state)

        max_per_fold = n_samples // self.n_splits

        groups = [[] for _ in range(self.n_splits)]

        def add_to_group(sample_idx, start_group, capacity):
            grp = start_group
            ctr = 0

            while ctr < self.n_splits:
                if len(groups[grp]) < capacity:
                    groups[grp].append(sample_idx)
                    return True

                grp = (grp + 1) % self.n_splits
                ctr += 1

            return False

        def divide2(start_idx, end_idx, capacity):
            idx = start_idx

            while idx < end_idx:
                grp = rng.randint(self.n_splits)

                if add_to_group(idx, grp, capacity):
                    idx += 1
                else:
                    return idx

            return -1

        # Phase 1: fill up to floor(n / folds)
        from_idx = divide2(0, n_samples, max_per_fold)

        # Phase 2: distribute leftovers
        if from_idx != -1:
            ok = divide2(from_idx, n_samples, max_per_fold + 1)

            if ok != -1:
                raise RuntimeError("Error partitioning xval data")

        fold_id = np.empty(n_samples, dtype=np.intp)

        for fold_idx, grp in enumerate(groups):
            for sample_idx in grp:
                fold_id[sample_idx] = fold_idx

        return fold_id

    def split(self, X, y=None, fold=None):
        n_samples = X.shape[0]
        fold_id = self.make_folds(n_samples)

        folds = range(self.n_splits) if fold is None else [int(fold)]

        for f in folds:
            if f < 0 or f >= self.n_splits:
                raise ValueError(
                    f"fold must be in [0, {self.n_splits - 1}], got {f}."
                )

            test_idx = np.flatnonzero(fold_id == f)
            train_idx = np.flatnonzero(fold_id != f)

            yield train_idx, test_idx


def _default_clus_metric(estimator, y_true, y_pred):
    """Default metric compatible with current PCTClassifier/PCTRegressor."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true_2d = y_true.reshape(-1, 1)
    else:
        y_true_2d = y_true

    if y_pred.ndim == 1:
        y_pred_2d = y_pred.reshape(-1, 1)
    else:
        y_pred_2d = y_pred

    # Classifier: higher is better.
    if hasattr(estimator, "classes_"):
        return accuracy_score(y_true_2d.ravel(), y_pred_2d.ravel())

    # Regressor: return negative MSE so higher is better.
    return -mean_squared_error(y_true_2d, y_pred_2d)


def clus_cross_validate(
    estimator,
    X,
    y,
    *,
    n_splits=10,
    random_state=0,
    fold=None,
    scoring=None,
    sample_weight=None,
    fit_params=None,
):
    """Run CLUS-style deterministic cross-validation."""
    X = np.asarray(X)
    y = np.asarray(y)

    if fit_params is None:
        fit_params = {}

    selector = ClusXValSelection(
        n_splits=n_splits,
        random_state=random_state,
    )

    scores = []
    estimators = []
    splits = []

    for train_idx, test_idx in selector.split(X, y, fold=fold):
        est = clone(estimator)

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        if sample_weight is None:
            est.fit(X_train, y_train, **fit_params)
        else:
            sw_train = np.asarray(sample_weight)[train_idx]

            est.fit(
                X_train,
                y_train,
                sample_weight=sw_train,
                **fit_params,
            )

        pred = est.predict(X_test)

        if scoring is None:
            score = _default_clus_metric(est, y_test, pred)
        else:
            score = scoring(est, X_test, y_test, pred)

        scores.append(float(score))
        estimators.append(est)
        splits.append((train_idx, test_idx))

    scores = np.asarray(scores, dtype=np.float64)

    return {
        "scores": scores,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "estimators": estimators,
        "splits": splits,
    }