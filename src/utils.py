import os
import json
import joblib


def create_output_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def print_separator():
    print("-" * 50)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_artifacts(model, scaler, output_folder):
    joblib.dump(model, os.path.join(output_folder, "model.joblib"))
    joblib.dump(scaler, os.path.join(output_folder, "scaler.joblib"))


def load_artifacts(output_folder):
    model = joblib.load(os.path.join(output_folder, "model.joblib"))
    scaler = joblib.load(os.path.join(output_folder, "scaler.joblib"))
    return model, scaler