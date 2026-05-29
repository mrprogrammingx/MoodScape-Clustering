from datetime import datetime
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def compute_cluster_metrics(X, labels):
    """
    Compute a small set of cluster quality metrics.

    Returns a dict with keys:
      - silhouette
      - davies_bouldin
      - calinski_harabasz
      - n_clusters
      - cluster_sizes (dict mapping cluster_id -> size)
    """
    metrics = {
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "n_clusters": None,
        "cluster_sizes": {},
    }

    if X is None or labels is None:
        return metrics

    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    metrics["n_clusters"] = int(len(unique))
    metrics["cluster_sizes"] = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}

    # Need at least 2 clusters and more than 1 sample per cluster for some metrics
    try:
        if metrics["n_clusters"] >= 2 and len(labels) > metrics["n_clusters"]:
            metrics["silhouette"] = float(silhouette_score(X, labels))
    except Exception:
        metrics["silhouette"] = None

    try:
        if metrics["n_clusters"] >= 2:
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception:
        metrics["davies_bouldin"] = None

    try:
        if metrics["n_clusters"] >= 2:
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception:
        metrics["calinski_harabasz"] = None

    metrics["computed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # cluster size anomaly detection
    try:
        total = int(len(labels))
        sizes = np.array(list(metrics["cluster_sizes"].values()))
        mean = float(sizes.mean()) if len(sizes) > 0 else 0.0
        std = float(sizes.std()) if len(sizes) > 0 else 0.0

        size_flags = {}
        # thresholds
        absolute_min = 3
        small_rel = 0.05  # <5% of total
        large_rel = 0.5   # >50% of total

        for cid, sz in metrics["cluster_sizes"].items():
            reason = []
            flag = "ok"
            if sz <= absolute_min:
                flag = "small"
                reason.append(f"size <= {absolute_min}")
            if total > 0 and (sz / total) < small_rel:
                flag = "small"
                reason.append(f"<{int(small_rel*100)}% of total")
            if len(sizes) > 1 and sz < (mean - 2 * std):
                flag = "small"
                reason.append("below mean - 2*std")
            if total > 0 and (sz / total) > large_rel:
                flag = "large"
                reason.append(f">{int(large_rel*100)}% of total")
            if len(sizes) > 1 and sz > (mean + 2 * std):
                flag = "large"
                reason.append("above mean + 2*std")

            size_flags[int(cid)] = {"flag": flag, "size": int(sz), "reasons": reason}

        metrics["size_flags"] = size_flags
        metrics["size_summary"] = {"total": total, "mean": mean, "std": std}
    except Exception:
        metrics["size_flags"] = {}
        metrics["size_summary"] = {}

    return metrics


def select_best_run(run_results, rule="highest_silhouette"):
    """
    Select the best run from a list of run_results.

    run_results: list of dicts each containing at least:
      - 'config'
      - 'metrics' (as returned by compute_cluster_metrics)
      - 'output_folder'

    rule: selection rule. Supported:
      - 'highest_silhouette' (default)
      - 'custom' (not implemented)

    Returns the selected run dict (one of the items from run_results).
    """
    if not run_results:
        return None

    if rule == "highest_silhouette":
        best = None
        best_score = float("-inf")
        for r in run_results:
            m = r.get("metrics", {})
            s = m.get("silhouette")
            ch = m.get("calinski_harabasz") or 0
            # prefer runs with available silhouette
            if s is None:
                score = -1e9 + ch / (1 + ch)
            else:
                # primary: silhouette, tie-breaker: calinski_harabasz
                score = s + (0.000001 * (ch or 0))

            if score > best_score:
                best_score = score
                best = r

        return best

    # fallback: return first
    return run_results[0]
