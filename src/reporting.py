import os
import zipfile


def build_zip_report(run_folder: str, out_folder="outputs/reports"):
    os.makedirs(out_folder, exist_ok=True)
    run_name = os.path.basename(run_folder.rstrip("/"))
    zip_path = os.path.join(out_folder, f"{run_name}_report.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fname in ["metrics.json", "cluster_summary.txt", "cluster_plot.png", "clustered_dataset.csv", "config_used.yaml"]:
            p = os.path.join(run_folder, fname)
            if os.path.exists(p):
                z.write(p, arcname=os.path.join(run_name, fname))
    return zip_path
