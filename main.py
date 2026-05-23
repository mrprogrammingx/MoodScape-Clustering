import argparse
import operator   

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

    parser.add_argument(
        "--scaler",
        choices=["standard", "minmax"],
        default="standard",
        help="Scaling method: standard (default) or minmax"
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated list of features to use, e.g. tempo,energy,danceability"
    )
    parser.add_argument(
        "--reduce-dims",
        dest="reduce_dims",
        action="store_true",
        default=False,
        help="Apply PCA before clustering"
    )
    parser.add_argument(
        "--n-components",
        dest="n_components",
        type=int,
        default=2,
        help="Number of PCA components (only used with --reduce-dims)"
    )
    parser.add_argument(
        "--min-k",
        dest="min_k",
        type=int,
        default=2,
        help="Minimum k when auto-selecting number of clusters"
    )
    parser.add_argument(
        "--max-k",
        dest="max_k",
        type=int,
        default=8,
        help="Maximum k when auto-selecting number of clusters"
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="'col op val'",
        help="Filter rows before clustering, e.g. --filter 'tempo > 100'. Can be repeated."
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

    # ---  apply CLI filters ---
    if args.filter:
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
        for expr in args.filter:
            parts = expr.strip().split()   # e.g. ["tempo", ">", "100"]
            col, op_str, val = parts[0], parts[1], float(parts[2])
            mask = mask & ops[op_str](df[col], val)
        df = df[mask].copy()
        print(f"[filter] {len(df)} rows remain after filtering.")


    print("Preprocessing...")

    # ---  build preprocessing config from CLI args ---
    features = [f.strip() for f in args.features.split(",")] if args.features else FEATURES
    preprocessing_cfg = {
        "scaler":       args.scaler,
        "features":     features,
        "reduce_dims":  args.reduce_dims,
        "n_components": args.n_components,
    }
    X_scaled, scaler = preprocess_data(df, config=preprocessing_cfg)

    if args.clusters:
        k = args.clusters
    else:
        k = find_best_k(X_scaled, min_k=args.min_k, max_k=args.max_k)   # added min_k/max_k

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
        features    # FEATURES — now uses whatever features were selected
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