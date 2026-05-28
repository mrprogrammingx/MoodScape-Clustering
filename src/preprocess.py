from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import joblib
import os
import json

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

def save_snapshot(snapshot_name, scaler, model, config, output_root="outputs"):
    """Save scaler, clustering model, and config metadata under a named folder."""
    folder = os.path.join(output_root, snapshot_name, "snapshot")
    os.makedirs(folder, exist_ok=True)
       
    joblib.dump(scaler, os.path.join(folder, "scaler.pk1"))
    joblib.dump(model, os.path.join(folder, "model.pk1")) #save clustering model
       
    #save config/metadata as json for reproducibility
    metadata = {
       "snapshot_name": snapshot_name,
       "preprocessing": config.get("preprocessing", {}),
       "clustering": config.get("clustering", {}),
       "filters": config.get("filters", []),
    }
    with open(os.path.join(folder, "metadata.json"), "w") as f:
       json.dump(metadata, f, indent=2)
        
    print(f"[snapshot] Saved to {folder}")

def load_snapshot(snapshot_name, output_root="outputs"):
    """Load a previously saved snapshot by name.
       Returns: scaler, model, metadata dict
    """
    folder = os.path.join(output_root, snapshot_name, "snapshot")
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"No snapshot found at {folder}")
    scaler = joblib.load(os.path.join(folder, "scaler.pk1"))
    model = joblib.load(os.path.join(folder, "model.pk1"))
    
    with open(os.path.join(folder, "metadata.json"), "r") as f:
        metadata = json.load(f)
    print(f"[snapshot] Loaded from {folder}")
    return scaler, model, metadata

    