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
