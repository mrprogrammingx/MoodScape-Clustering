import argparse

from src.data_loader import load_dataset
from src.preprocess import preprocess_data
from src.clustering import (
    find_best_k,
    cluster_data
)
from src.visualization import visualize_clusters
from src.summarizer import (
    summarize_clusters,
    generate_mood_profile
)
from src.utils import (
    create_output_folder,
    save_text
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

    parser.add_argument(
        "--input",
        required=False,
        help="Path to dataset"
    )

    parser.add_argument(
        "--output",
        default="outputs"
    )

    parser.add_argument(
        "--clusters",
        type=int
    )

    parser.add_argument(
        "--config",
        help="Path to YAML config file describing one or more runs"
    )

    args = parser.parse_args()

    # If a config file is provided, run batch jobs
    if args.config:
        from src.runner import run_from_config_file
        run_from_config_file(args.config)
        return

    # legacy single-run behavior
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

    summaries = summarize_clusters(
        df,
        FEATURES
    )

    summary_text = ""

    for cluster_id, stats in summaries.items():

        mood = generate_mood_profile(stats)

        summary_text += (
            f"Cluster {cluster_id}\n"
        )

        summary_text += (
            f"Size: {stats['size']}\n"
        )

        summary_text += (
            f"Mood: {mood}\n\n"
        )

    save_text(
        summary_text,
        f"{args.output}/cluster_summary.txt"
    )

    df.to_csv(
        f"{args.output}/clustered_dataset.csv",
        index=False
    )

    print("Done!")


if __name__ == "__main__":
    main()
