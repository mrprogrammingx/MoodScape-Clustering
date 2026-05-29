import os
import shutil
import json


def export_bundle(run_folders, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    manifest = {"runs": []}
    for run in run_folders:
        name = os.path.basename(run.rstrip("/"))
        dest_run = os.path.join(dest_folder, name)
        if os.path.exists(dest_run):
            shutil.rmtree(dest_run)
        shutil.copytree(run, dest_run)
        manifest["runs"].append({"name": name, "path": dest_run})

    # also try to include global history if present
    history_src = os.path.join("outputs", "run_history.json")
    if os.path.exists(history_src):
        shutil.copy(history_src, os.path.join(dest_folder, "run_history.json"))
        manifest["history"] = "run_history.json"

    manifest_path = os.path.join(dest_folder, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return dest_folder
