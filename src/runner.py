import os
import operator    
import yaml
from src.data_loader import load_dataset
from src.preprocess import preprocess_data
from src.clustering import find_best_k, cluster_data
from src.visualization import visualize_clusters
from src.summarizer import summarize_clusters, generate_mood_profile
from src.utils import create_output_folder, save_text, save_json


# ---  filter helper -------------------------------------------
def apply_filters(df, filters):
    """
    filters is a list of dicts like:
      {"column": "tempo", "op": ">", "value": 100}
    Supported ops: >  >=  <  <=  ==  !=
    """
    if not filters:
        return df

    import pandas as pd
    ops = {
        ">":  operator.gt,
        ">=": operator.ge,
        "<":  operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    mask = pd.Series([True] * len(df), index=df.index)
    for f in filters:
        col, op_str, val = f["column"], f["op"], float(f["value"])
        mask = mask & ops[op_str](df[col], val)

    filtered = df[mask].copy()
    print(f"[filter] {len(df) - len(filtered)} rows removed, {len(filtered)} remain.")
    return filtered


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

    # ---  apply filters before preprocessing ---
    filters = config.get("filters", [])
    df = apply_filters(df, filters)

    print("Preprocessing...")
    # ---  pass preprocessing config ---
    preprocessing_cfg = config.get("preprocessing", {})
    X_scaled, scaler = preprocess_data(df, config=preprocessing_cfg)
    
    k = None
    if config.get("clustering") and config["clustering"].get("k"):
        k = config["clustering"]["k"]

    # ---  support min_k / max_k from config ---
    min_k = config.get("clustering", {}).get("min_k", 2)
    max_k = config.get("clustering", {}).get("max_k", 8)

    if k is None:
        k = find_best_k(X_scaled, min_k=min_k, max_k=max_k)

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