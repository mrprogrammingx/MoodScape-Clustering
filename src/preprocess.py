from sklearn.preprocessing import StandardScaler

FEATURES = [
    "tempo",
    "energy",
    "danceability",
    "skips",
    "replays"
]

def preprocess_data(df):
    X = df[FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler
