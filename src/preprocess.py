from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

FEATURES = [
    "tempo",
    "energy",
    "danceability",
    "skips",
    "replays"
]

def preprocess_data(df, config=None):      
   
    if config is None:
        config = {}

    scaler_name  = config.get("scaler", "standard")   
    features     = config.get("features", FEATURES)
    reduce_dims  = config.get("reduce_dims", False)
    n_components = config.get("n_components", 2)


    X = df[features]                       

    if scaler_name == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()          

    X_scaled = scaler.fit_transform(X)

    pca = None
    if reduce_dims:
        n_components = min(n_components, X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"[preprocess] PCA applied: {n_components} components, "
              f"variance explained: {pca.explained_variance_ratio_.sum():.1%}")


    return X_scaled, scaler               