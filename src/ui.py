import os
import platform
import subprocess
from src.runner import run_single_config
from src.compare import compare_two_runs, save_comparison
from src.reporting import build_zip_report
from src.evidence import export_bundle
import json

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

    # Offer a simple post-run menu for comparison/export
    print("\nAvailable actions:")
    print("  [1] Compare two existing runs")
    print("  [2] Build a zip report for a run")
    print("  [3] Export an evidence bundle for selected runs")
    print("  [Enter] Skip")
    choice = input("Select action: ")
    if choice.strip() == "1":
        outputs_dir = "outputs"
        runs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
        print("\nAvailable run folders:")
        for i, r in enumerate(runs):
            print(f"  [{i + 1}] {r}")
        try:
            a = int(input("Select first run number: ")) - 1
            b = int(input("Select second run number: ")) - 1
            run_a = os.path.join(outputs_dir, runs[a])
            run_b = os.path.join(outputs_dir, runs[b])
        except Exception:
            print("Invalid selection. Skipping.")
            return
        print(f"Comparing {runs[a]} vs {runs[b]}...")
        try:
            result = compare_two_runs(run_a, run_b)
            cmp_folder = os.path.join("outputs", "comparisons")
            os.makedirs(cmp_folder, exist_ok=True)
            out_path = os.path.join(cmp_folder, f"cmp_{runs[a]}_vs_{runs[b]}.json")
            save_comparison(result, out_path)
            print(f"Comparison saved to {out_path}")
        except Exception as e:
            print(f"Comparison failed: {e}")

    elif choice.strip() == "2":
        outputs_dir = "outputs"
        runs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
        print("\nAvailable run folders:")
        for i, r in enumerate(runs):
            print(f"  [{i + 1}] {r}")
        try:
            sel = int(input("Select run number to build report for: ")) - 1
            run_folder = os.path.join(outputs_dir, runs[sel])
        except Exception:
            print("Invalid selection. Skipping.")
            return
        zip_path = build_zip_report(run_folder)
        print(f"Report created: {zip_path}")

    elif choice.strip() == "3":
        outputs_dir = "outputs"
        runs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
        print("\nAvailable run folders:")
        for i, r in enumerate(runs):
            print(f"  [{i + 1}] {r}")
        sel = input("Enter run numbers to include (comma-separated), e.g. 1,3: ")
        try:
            idxs = [int(s.strip()) - 1 for s in sel.split(",")]
            run_folders = [os.path.join(outputs_dir, runs[i]) for i in idxs]
        except Exception:
            print("Invalid selection. Skipping.")
            return
        dest = input("Enter destination folder for evidence bundle (defaults to outputs/evidence_bundle): ")
        if not dest:
            dest = os.path.join("outputs", "evidence_bundle")
        export_folder = export_bundle(run_folders, dest)
        print(f"Evidence bundle exported to {export_folder}")