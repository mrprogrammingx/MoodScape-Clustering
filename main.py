import argparse
import operator   

from src.data_loader import load_dataset
from src.preprocess import preprocess_data, save_snapshot
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
   
    parser.add_argument("--predict", help="Path to new data for prediction")
    parser.add_argument("--inspect", type=int, help="Cluster ID to deeply inspect")
    parser.add_argument(
        "--save-snapshot",
        dest="save_snapshot",
        default=None,
        help="Save the scaler and model under this snapshot name, e.g. --save-snapshot my_run"
    )
        
       

    args = parser.parse_args()

    # If no input or config is provided, launch interactive CLI UI
    if not args.input and not args.config:
        from src.ui import run_cli_ui
        run_cli_ui()
        return

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

    distinguishing = get_distinguishing_features(df, features)


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

        profile = generate_cluster_profile(df, features, cluster_id)
        top_features = distinguishing.get(cluster_id, [])
        summary_text += profile + "\n"
        summary_text += f"Top features: {', '.join(top_features)}\n\n"

    save_text(
        summary_text,
        f"{args.output}/cluster_summary.txt"
    )
    # append size anomaly warnings if available
    try:
        import json
        with open(f"{args.output}/metrics.json", "r", encoding="utf-8") as mf:
            m = json.load(mf)
            size_flags = m.get("size_flags", {})
            if size_flags:
                extra = "Cluster size checks:\n"
                for cid, info in size_flags.items():
                    if info["flag"] != "ok":
                        extra += f" - Cluster {cid}: {info['flag']} ({'; '.join(info['reasons'])}) size={info['size']}\n"
                with open(f"{args.output}/cluster_summary.txt", "a", encoding="utf-8") as f:
                    f.write("\n" + extra)
    except Exception:
        pass
    # Deep inspect a specific cluster if requested
    if args.inspect is not None:
        inspect_cluster(df, features, args.inspect)

    # Export summaries and representative samples
    export_cluster_summary(df, features, f"{args.output}/cluster_stats.csv")
    export_representative_samples(df, features, f"{args.output}/representative_samples.csv")

    df.to_csv(
        f"{args.output}/clustered_dataset.csv",
        index=False
    )
    save_artifacts(model, scaler, args.output)

    # Save snapshot if name provided
    if args.save_snapshot:
        preprocessing_cfg = {
            "scaler": args.scaler,
            "features": features,
            "reduce_dims": args.reduce_dims,
            "n_components": args.n_components,
        }
        full_config = {
            "preprocessing": preprocessing_cfg,
            "clustering": {"k": k, "min_k": args.min_k, "max_k": args.max_k},
            "filters": [],
        }
        save_snapshot(args.save_snapshot, scaler, model, full_config, output_root=args.output)
    print("Done!")


if __name__ == "__main__":
    main()