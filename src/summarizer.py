import pandas as pd


def summarize_clusters(df, features):
    summaries = {}

    grouped = df.groupby("cluster")

    for cluster_id, group in grouped:
        summaries[cluster_id] = {
            "size": len(group),
            "means": group[features].mean().to_dict(),
            "medians": group[features].median().to_dict()
        }

    return summaries


def generate_mood_profile(stats):
    means = stats["means"]

    mood_parts = []

    if means["energy"] > 0.7:
        mood_parts.append("high-energy")

    if means["danceability"] > 0.7:
        mood_parts.append("dance-focused")

    if means["tempo"] < 90:
        mood_parts.append("slow-paced")

    if means["tempo"] > 130:
        mood_parts.append("fast-paced")

    if means["replays"] > means["skips"]:
        mood_parts.append("high replay value")

    if means["skips"] > means["replays"]:
        mood_parts.append("frequently skipped")

    if not mood_parts:
        mood_parts.append("balanced")

    return ", ".join(mood_parts)

def get_distinguishing_features(df, features, top_n=3):
    """
    For each cluster, rank features by how much they deviate
    from the global mean (relative difference).
    """
    global_means = df[features].mean()
    results = {}
    for cluster_id in df["cluster"].unique():
        cluster_df = df[df["cluster"] == cluster_id]
        cluster_means = cluster_df[features].mean()
        relative_diff = ((cluster_means - global_means) / global_means).abs()
        top_features = relative_diff.sort_values(ascending=False).head(top_n)
        results[cluster_id] = top_features.index.tolist()
    return results


def inspect_cluster(df, features, cluster_id, n_samples=5):
    """
    Deep inspection of a single cluster:
    - Full statistics (mean, std, min, max)
    - Representative data points closest to the cluster centroid
    """
    cluster_df = df[df["cluster"] == cluster_id][features]

    stats = cluster_df.describe().round(3)

    centroid = cluster_df.mean()
    distances = cluster_df.apply(lambda row: ((row - centroid) ** 2).sum() ** 0.5, axis=1)
    representative = df.loc[distances.nsmallest(n_samples).index]

    print(f"\n── Cluster {cluster_id} Inspection ──")
    print(f"Size: {len(cluster_df)}")
    print("\nStatistics:")
    print(stats.to_string())
    print(f"\nTop {n_samples} representative samples:")
    print(representative[features].to_string())

    return stats, representative


def generate_cluster_profile(df, features, cluster_id):
    """
    Generate a human-readable text description of a cluster
    based on its most distinguishing features.
    """
    cluster_df = df[df["cluster"] == cluster_id]
    global_means = df[features].mean()
    cluster_means = cluster_df[features].mean()

    lines = []
    for feature in features:
        diff = cluster_means[feature] - global_means[feature]
        pct = (diff / global_means[feature]) * 100
        direction = "higher" if diff > 0 else "lower"
        if abs(pct) >= 10:
            lines.append(f"  - {feature} is {abs(pct):.0f}% {direction} than average")

    mood = generate_mood_profile({"means": cluster_means.to_dict()})
    profile = f"Cluster {cluster_id} ({len(cluster_df)} songs) — {mood}\n"
    profile += "\n".join(lines) if lines else "  - Close to average across all features"

    return profile


def export_cluster_summary(df, features, output_path):
    """
    Save per-cluster mean statistics to a CSV.
    """
    summary = df.groupby("cluster")[features].mean().round(3)
    summary["size"] = df.groupby("cluster").size()
    summary.to_csv(output_path)
    print(f"Cluster summary exported to {output_path}")


def export_representative_samples(df, features, output_path, n_samples=5):
    """
    For each cluster, pick the n samples closest to the centroid
    and save them all to a single CSV.
    """
    reps = []
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        centroid = cluster_df[features].mean()
        distances = cluster_df[features].apply(
            lambda row: ((row - centroid) ** 2).sum() ** 0.5, axis=1
        )
        reps.append(df.loc[distances.nsmallest(n_samples).index])

    pd.concat(reps).to_csv(output_path, index=False)
    print(f"Representative samples exported to {output_path}")