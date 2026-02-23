import numpy as np
import pytest
from sklearn.tree import PCTClassifier


def test_clus_modified_entropy_is_not_supported_anymore():
    X = np.array([[0.0], [1.0]], dtype=np.float32)
    y = np.array([[0.0], [1.0]], dtype=float)

    with pytest.raises(Exception) as excinfo:
        PCTClassifier(
            criterion="clus_modified_entropy",
            max_depth=1,
            random_state=0,
            missing_target_attr_handling="parent_node",
        ).fit(X, y)

    msg = str(excinfo.value).lower()
    assert "clus_modified_entropy" in msg