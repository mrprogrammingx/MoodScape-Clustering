import json
import os
from datetime import datetime

def save_run_history(config, metrics, output_folder, history_path="outputs/run_history.json"):
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    history_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "metrics": metrics,
        "output_folder": output_folder
    }
    
    history = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                pass
                
    history.append(history_record)
    
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


def save_best_run(best_summary, path="outputs/best_run.json"):
        """
        Save a concise summary of the selected best run.

        best_summary should be a dict with keys like:
            - selected_by
            - best_output_folder
            - best_metrics
            - best_config
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
                json.dump(best_summary, f, indent=2)