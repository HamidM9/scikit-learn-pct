import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    hamming_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _fmt_e(x):
    return f"{float(x):.4E}".replace("E+0", "E").replace("E-0", "E-")


def clus_model_info(estimator):
    """Return CLUS-style tree model information."""
    return {
        "nodes": int(estimator.tree_.node_count),
        "leaves": int(estimator.tree_.n_leaves),
        "depth": int(estimator.tree_.max_depth),
        "clus_depth": int(estimator.tree_.max_depth + 1),
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


def clus_tree_text(estimator, feature_names=None):
    """Return a simple CLUS-like text representation of a fitted tree."""
    tree = estimator.tree_

    if feature_names is None:
        feature_names = [f"x{i + 1}" for i in range(estimator.n_features_in_)]

    lines = []

    def recurse(node_id, prefix=""):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        if left == -1 and right == -1:
            lines.append(prefix + "Leaf")
            return

        feature = tree.feature[node_id]
        name = feature_names[feature]

        lines.append(prefix + f"{name} = 0")
        recurse(left, prefix + "|   ")

        lines.append(prefix + f"{name} = 1")
        recurse(right, prefix + "|   ")

    recurse(0)
    return "\n".join(lines)


def clus_classification_text(estimator, X, y, *, target_names=None):
    """Return CLUS-like classification training-error text."""
    y_true = np.asarray(y)
    y_pred = np.asarray(estimator.predict(X))

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_samples, n_outputs = y_true.shape
    n_observed_examples = int(np.sum(~np.any(np.isnan(y_true), axis=1)))

    if target_names is None:
        target_names = [f"y{i + 1}" for i in range(n_outputs)]

    positive_rate = np.mean(y_true == 1, axis=0)
    default_labels = (positive_rate >= 0.5).astype(int)
    default_pred = np.tile(default_labels, (n_samples, 1))

    lines = []
    lines.append("")
    lines.append("Training error")
    lines.append("--------------")
    lines.append("")
    lines.append(f"Number of examples: {n_samples}")
    lines.append("Classification Error")

    blocks = [
        ("Default", default_pred),
        ("Original", y_pred),
    ]

    for block_name, block_pred in blocks:
        lines.append(f"   {block_name}: ")

        for k, name in enumerate(target_names):
            observed = ~np.isnan(y_true[:, k])

            yt = y_true[observed, k].astype(int)
            yp = block_pred[observed, k].astype(int)

            cm = confusion_matrix(yt, yp, labels=[1, 0])
            acc = accuracy_score(yt, yp)

            row_totals = cm.sum(axis=1)
            col_totals = cm.sum(axis=0)
            total = cm.sum()

            lines.append(f"   Attribute: {name}")
            lines.append("     REAL\\PRED | 1 | 0 |")
            lines.append("     ---------------------")
            lines.append(f"             1 | {cm[0, 0]} | {cm[0, 1]} | {row_totals[0]}")
            lines.append(f"             0 | {cm[1, 0]} | {cm[1, 1]} | {row_totals[1]}")
            lines.append("     ---------------------")
            lines.append(f"               | {col_totals[0]} | {col_totals[1]} | {total}")
            lines.append(f"     Accuracy: {_fmt_e(acc)}")
            lines.append("")

    return "\n".join(lines)


def clus_full_classification_output(
    estimator,
    X,
    y,
    *,
    feature_names=None,
    target_names=None,
):
    """Return CLUS-like full classification output as a string."""
    info = clus_model_info(estimator)

    parts = []

    parts.append("")
    parts.append("CLUS-like tree")
    parts.append(clus_tree_text(estimator, feature_names=feature_names))

    parts.append("")
    parts.append("Model information")
    parts.append(f"Nodes:  {info['nodes']}")
    parts.append(f"Leaves: {info['leaves']}")
    parts.append(f"Depth:  {info['clus_depth']}")

    parts.append(
        clus_classification_text(
            estimator,
            X,
            y,
            target_names=target_names,
        )
    )

    return "\n".join(parts)


def save_clus_full_classification_output(
    estimator,
    X,
    y,
    output_path,
    *,
    feature_names=None,
    target_names=None,
):
    """Save CLUS-like full classification output to a text file."""
    text = clus_full_classification_output(
        estimator,
        X,
        y,
        feature_names=feature_names,
        target_names=target_names,
    )

    with open(output_path, "w") as f:
        f.write(text)

    return output_path