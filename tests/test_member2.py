"""Tests for preprocessing consistency and snapshot save/load"""
import os
import shutil
import pandas as pd
import numpy as np
from src.preprocess import preprocess_data, save_snapshot, load_snapshot
from sklearn.cluster import KMeans

def make_fake_df():
    np.random.seed(42)
    return pd.DataFrame({
        "tempo":        np.random.uniform(80, 160, 50),
        "energy":       np.random.uniform(0, 1, 50),
        "danceability": np.random.uniform(0, 1, 50),  
        "skips":        np.random.uniform(0, 50, 50),
        "replays":      np.random.uniform(0, 50, 50)
    })

# Gives zero mean
def test_standard_scaler_zero_mean():
    df = make_fake_df()
    X_scaled, scaler = preprocess_data(df, config={"scaler": "standard"})
    means = X_scaled.mean(axis=0)
    assert np.allclose(means, 0, atol=1e-6), "Standard scaler should produce zero mean"
    print("Pass: test_standard_scaler_zero_mean")

# minmax scaler gives values between 0 and 1
def test_minmax_scaler_range():
    df = make_fake_df()
    X_scaled, scaler = preprocess_data(df, config={"scaler": "minmax"})
    assert X_scaled.min() >= 0.0, "MinMax scaler min should be >= 0"
    assert X_scaled.max() <= 1.0 + 1e-9, "MinMax scaler max should be <= 1"
    print("Pass: test_minmax_scaler_range")

# Feature selection uses only chosen features
def test_feauture_selection():
    df = make_fake_df()
    chosen = ["tempo", "energy"]
    X_scaled, scaler = preprocess_data(df, config={"features": chosen})
    assert X_scaled.shape[1] == 2, "Should only have 2 columns when 2 features selected"  # fixed: was X_scaled[1]
    print("Pass: test_feature_selection")

def test_pca_reduces_dims():
    df = make_fake_df()
    X_scaled, scaler = preprocess_data(df, config={"reduce_dims": True, "n_components": 2})
    assert X_scaled.shape[1] == 2, "PCA should reduce to 2 components"
    print("Pass: test_pca_reduces_dims")

def test_snapshot_save_and_load():
    df = make_fake_df()
    config = {
        "preprocessing": {"scaler": "standard"},
        "clustering":    {"k": 3},
        "filters":       [],
    }

    X_scaled, scaler = preprocess_data(df, config=config["preprocessing"])
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X_scaled)

    test_output = "outputs/test_snapshots"
    save_snapshot("test_run", scaler, model, config, output_root=test_output)

    loaded_scaler, loaded_model, metadata = load_snapshot("test_run", output_root=test_output)

    assert metadata["snapshot_name"] == "test_run"
    assert metadata["clustering"]["k"] == 3

    X_original = X_scaled
    X_reloaded = loaded_scaler.transform(df[["tempo", "energy", "danceability", "skips", "replays"]])
    assert np.allclose(X_original, X_reloaded, atol=1e-6), "Loaded scaler should give same output"

    print("PASS: test_snapshot_save_and_load")

    shutil.rmtree(test_output)

if __name__ == "__main__":
    test_standard_scaler_zero_mean()
    test_minmax_scaler_range()
    test_feauture_selection()
    test_pca_reduces_dims()
    test_snapshot_save_and_load()
    print("\nAll Member 2 tests passed.")