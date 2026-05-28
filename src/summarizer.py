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
    global_means = df[features].mean()
    results = {}
    for cluster_id in df["cluster"].unique():
        cluster_df = df[df["cluster"] == cluster_id]
        cluster_means = cluster_df[features].mean()
        relative_diff = ((cluster_means - global_means) / global_means).abs()
        top_features = relative_diff.sort_values(ascending=False).head(top_n)
        results[cluster_id] = top_features.index.tolist()
    return results