import numpy as np

from sklearn.metrics import accuracy_score, hamming_loss, mean_absolute_error, mean_squared_error, r2_score


def clus_model_info(estimator):
    """Return CLUS-style tree model information."""
    return {
        "nodes": int(estimator.tree_.node_count),
        "leaves": int(estimator.tree_.n_leaves),
        "depth": int(estimator.tree_.max_depth),
    }


def clus_classification_report(estimator, X, y):
    """Return CLUS-style classification summary metrics."""
    y_true = np.asarray(y)
    y_pred = np.asarray(estimator.predict(X))

    if y_true.ndim == 1:
        y_true_2d = y_true.reshape(-1, 1)
    else:
        y_true_2d = y_true

    if y_pred.ndim == 1:
        y_pred_2d = y_pred.reshape(-1, 1)
    else:
        y_pred_2d = y_pred

    return {
        "accuracy": float(accuracy_score(y_true_2d.ravel(), y_pred_2d.ravel())),
        "subset_accuracy": float(accuracy_score(y_true_2d, y_pred_2d)),
        "hamming_loss": float(hamming_loss(y_true_2d, y_pred_2d)),
    }


def clus_regression_report(estimator, X, y):
    """Return CLUS-style regression summary metrics."""
    y_true = np.asarray(y, dtype=np.float64)
    y_pred = np.asarray(estimator.predict(X), dtype=np.float64)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def clus_report(estimator, X, y):
    """Return compact CLUS-style report for a fitted estimator."""
    report = {
        "model": clus_model_info(estimator),
    }

    if hasattr(estimator, "classes_"):
        report["classification"] = clus_classification_report(estimator, X, y)
    else:
        report["regression"] = clus_regression_report(estimator, X, y)

    return report