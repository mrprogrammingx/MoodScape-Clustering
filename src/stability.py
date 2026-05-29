import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.compare import compare_two_runs


def compute_pairwise_ari(run_folders):
    n = len(run_folders)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                res = compare_two_runs(run_folders[i], run_folders[j])
                matrix[i, j] = res.get("ari", 0.0)
                matrix[j, i] = matrix[i, j]
    return matrix


def save_stability_report(run_folders, out_folder="outputs/stability"):
    os.makedirs(out_folder, exist_ok=True)
    matrix = compute_pairwise_ari(run_folders)
    names = [os.path.basename(p) for p in run_folders]

    # Save numeric report
    report = {
        "runs": names,
        "pairwise_ari": matrix.tolist(),
        "mean_ari": float(np.triu(matrix, k=1).mean()) if len(run_folders) > 1 else 1.0,
    }
    with open(os.path.join(out_folder, "stability_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(label="ARI")
    plt.xticks(range(len(names)), names, rotation=45, fontsize=8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "pairwise_ari.png"))
    plt.close()

    return os.path.join(out_folder, "stability_report.json")
