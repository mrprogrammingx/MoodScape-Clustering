import os
import json
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _load_clustered_csv(run_folder):
    path = os.path.join(run_folder, "clustered_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"clustered_dataset.csv not found in {run_folder}")
    return pd.read_csv(path)


def compare_two_runs(run_a: str, run_b: str):
    """
    Compare cluster assignments between two run folders.

    Returns a dict with ARI, NMI and overlap table and counts.
    """
    a = _load_clustered_csv(run_a)
    b = _load_clustered_csv(run_b)

    # Align by common id if present
    merge_on = None
    for col in ["id", "track_id", "song_id"]:
        if col in a.columns and col in b.columns:
            merge_on = col
            break

    if merge_on:
        merged = a.merge(b[[merge_on, "cluster"]].rename(columns={"cluster": "cluster_b"}), on=merge_on)
        labels_a = merged["cluster"].values
        labels_b = merged["cluster_b"].values
    else:
        # fallback: align by index
        min_len = min(len(a), len(b))
        labels_a = a["cluster"].values[:min_len]
        labels_b = b["cluster"].values[:min_len]

    ari = float(adjusted_rand_score(labels_a, labels_b))
    nmi = float(normalized_mutual_info_score(labels_a, labels_b))

    overlap = pd.crosstab(labels_a, labels_b)

    result = {
        "ari": ari,
        "nmi": nmi,
        "overlap_table": overlap.to_dict(),
        "counts_a": a["cluster"].value_counts().to_dict(),
        "counts_b": b["cluster"].value_counts().to_dict(),
    }

    return result


def save_comparison(result: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return out_path
