import os
import yaml
from src.data_loader import load_dataset
from src.preprocess import preprocess_data
from src.clustering import find_best_k, cluster_data
from src.visualization import visualize_clusters
from src.summarizer import summarize_clusters, generate_mood_profile
from src.utils import create_output_folder, save_text, save_json


def run_single_config(config):
    dataset = config.get("dataset")
    output_root = config.get("output_root", "outputs")
    output_slug = config.get("output_slug", config.get("run_name", "run"))

    run_folder = os.path.join(output_root, f"{output_slug}")
    create_output_folder(run_folder)

    # Save config
    with open(os.path.join(run_folder, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    print(f"Loading dataset {dataset}...")
    df = load_dataset(dataset)

    print("Preprocessing...")
    X_scaled, scaler = preprocess_data(df)

    k = None
    if config.get("clustering") and config["clustering"].get("k"):
        k = config["clustering"]["k"]

    if k is None:
        k = find_best_k(X_scaled)

    print(f"Using K={k}")
    model, labels = cluster_data(X_scaled, k)

    df["cluster"] = labels

    print("Generating visualization...")
    visualize_clusters(X_scaled, labels, os.path.join(run_folder, "cluster_plot.png"))

    print("Generating summaries...")
    features = config.get("features", ["tempo", "energy", "danceability", "skips", "replays"])
    summaries = summarize_clusters(df, features)

    summary_text = ""
    for cluster_id, stats in summaries.items():
        mood = generate_mood_profile(stats)
        summary_text += (f"Cluster {cluster_id}\n")
        summary_text += (f"Size: {stats['size']}\n")
        summary_text += (f"Mood: {mood}\n\n")

    save_text(summary_text, os.path.join(run_folder, "cluster_summary.txt"))
    df.to_csv(os.path.join(run_folder, "clustered_dataset.csv"), index=False)

    metrics = {"k": k}
    save_json(metrics, os.path.join(run_folder, "metrics.json"))

    print(f"Run saved to {run_folder}")


def run_from_config_file(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # support either a top-level 'runs' list or a single config
    if isinstance(cfg, dict) and "runs" in cfg:
        runs = cfg["runs"]
    else:
        runs = [cfg]

    for run_cfg in runs:
        run_single_config(run_cfg)
