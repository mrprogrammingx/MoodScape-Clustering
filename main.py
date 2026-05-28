import argparse

from src.data_loader import load_dataset
from src.preprocess import preprocess_data
from src.clustering import (
    find_best_k,
    cluster_data,
    predict_clusters
)
from src.visualization import visualize_clusters
from src.summarizer import (
    summarize_clusters,
    generate_mood_profile,
    get_distinguishing_features,
    generate_cluster_profile,
    inspect_cluster,
    export_cluster_summary,
    export_representative_samples
)
from src.utils import (
    create_output_folder,
    save_text,
    save_artifacts,
    load_artifacts
)


FEATURES = [
    "tempo",
    "energy",
    "danceability",
    "skips",
    "replays"
]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=False, help="Path to dataset")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--clusters", type=int)
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--predict", help="Path to new data for prediction")
    parser.add_argument("--inspect", type=int, help="Cluster ID to deeply inspect")

    args = parser.parse_args()

    # If a config file is provided, run batch jobs
    if args.config:
        from src.runner import run_from_config_file
        run_from_config_file(args.config)
        return

    # Prediction mode
    if args.predict:
        model, scaler = load_artifacts(args.output)
        new_df = load_dataset(args.predict)
        predict_clusters(new_df, model, scaler, FEATURES, f"{args.output}/predictions.csv")
        print(f"Predictions saved to {args.output}/predictions.csv")
        return

    # Normal run
    create_output_folder(args.output)

    print("Loading dataset...")
    df = load_dataset(args.input)


    print("Preprocessing...")
    X_scaled, scaler = preprocess_data(df)

    if args.clusters:
        k = args.clusters
    else:
        k = find_best_k(X_scaled)

    print(f"Using K={k}")

    model, labels = cluster_data(X_scaled, k)

    df["cluster"] = labels

    print("Generating visualization...")
    visualize_clusters(
        X_scaled,
        labels,
        f"{args.output}/cluster_plot.png"
    )

    print("Generating summaries...")
    summaries = summarize_clusters(df, FEATURES)
    distinguishing = get_distinguishing_features(df, FEATURES)

    summary_text = ""
    for cluster_id, stats in summaries.items():
        profile = generate_cluster_profile(df, FEATURES, cluster_id)
        top_features = distinguishing.get(cluster_id, [])
        summary_text += profile + "\n"
        summary_text += f"Top features: {', '.join(top_features)}\n\n"

    save_text(summary_text, f"{args.output}/cluster_summary.txt")

    # Deep inspect a specific cluster if requested
    if args.inspect is not None:
        inspect_cluster(df, FEATURES, args.inspect)

    # Export summaries and representative samples
    export_cluster_summary(df, FEATURES, f"{args.output}/cluster_stats.csv")
    export_representative_samples(df, FEATURES, f"{args.output}/representative_samples.csv")

    df.to_csv(
        f"{args.output}/clustered_dataset.csv",
        index=False
    )
    save_artifacts(model, scaler, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
