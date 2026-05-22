import pandas as pd


def load_dataset(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
    "feature_1": "tempo",
    "feature_2": "energy",
    "feature_3": "danceability",
    "feature_4": "skips",
    "feature_5": "replays"
})
    df = df.dropna()

    
    for col in df.columns:
        df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
        df[col] = df[col].astype(float)

    return df
