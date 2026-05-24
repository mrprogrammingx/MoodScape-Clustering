import os
import platform
import subprocess
from src.runner import run_single_config

def run_cli_ui():
    print("=" * 50)
    print("MoodScape Clustering UI")
    print("=" * 50)
    
    data_dir = "data"
    datasets = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    
    if not datasets:
        print("Error: No datasets found in 'data/' folder.")
        return
        
    print("\nAvailable datasets:")
    for i, ds in enumerate(datasets):
        print(f"  [{i + 1}] {ds}")
        
    try:
        ds_choice = int(input("\nSelect dataset number: ")) - 1
        selected_dataset = os.path.join(data_dir, datasets[ds_choice])
    except (ValueError, IndexError):
        print("Invalid choice. Exiting.")
        return
        
    k_input = input("\nEnter number of clusters (leave blank for auto-detect): ")
    k = int(k_input) if k_input.strip().isdigit() else None
    
    run_name = input("\nEnter run name (e.g., run_01): ").strip()
    if not run_name:
        run_name = "interactive_run"
        
    config = {
        "run_name": run_name,
        "output_slug": run_name,
        "dataset": selected_dataset,
        "clustering": {}
    }
    if k:
        config["clustering"]["k"] = k
        
    print(f"\nRunning clustering pipeline on {selected_dataset}...")
    run_single_config(config)
    
    run_folder = os.path.join("outputs", run_name)
    summary_path = os.path.join(run_folder, "cluster_summary.txt")
    plot_path = os.path.join(run_folder, "cluster_plot.png")
    
    if os.path.exists(summary_path):
        view_summary = input("\nView cluster summary now? (y/n): ")
        if view_summary.lower() == 'y':
            print("\n" + "="*50)
            with open(summary_path, "r", encoding="utf-8") as f:
                print(f.read())
            print("="*50)
            
    if os.path.exists(plot_path):
        view_plot = input("\nOpen cluster plot image? (y/n): ")
        if view_plot.lower() == 'y':
            print("Opening plot...")
            system = platform.system()
            try:
                if system == 'Windows':
                    os.startfile(plot_path)
                elif system == 'Darwin':
                    subprocess.run(['open', plot_path])
                else:
                    subprocess.run(['xdg-open', plot_path])
            except Exception as e:
                print(f"Could not open file automatically: {e}")
    
    print(f"\nDone! Files saved in '{run_folder}'.")