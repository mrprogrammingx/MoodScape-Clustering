import numpy as np
from src.metrics import compute_cluster_metrics


def test_size_flags_small_and_large():
    # create synthetic labels with an outlier small cluster and a large cluster
    labels = np.array([0] * 50 + [1] * 2 + [2] * 5)
    X = np.random.RandomState(42).rand(len(labels), 3)
    metrics = compute_cluster_metrics(X, labels)
    assert "size_flags" in metrics
    flags = metrics["size_flags"]
    assert flags[1]["flag"] == "small"
    assert flags[0]["flag"] == "large" or flags[0]["flag"] == "ok"


if __name__ == "__main__":
    test_size_flags_small_and_large()
    print("metrics tests passed")
