in cmd / terminal type
python main.py --input data/dataset_A.csv
or
python main.py --input data/dataset_B.csv

the output is visible in the outputs folder

Keywords: music, clustering, mood

For deep inspection of a specific cluster:
python main.py --input data/dataset_A.csv --inspect 0

For prediction on new data:
python main.py --predict data/dataset_B.csv

For run based on config file, you can run this:
python3 main.py --config configs/example_runs.yaml

## Virtual environment (recommended)

From the project root, create and activate a project-local virtual environment and install dependencies:

# MoodScape-Clustering

MoodScape-Clustering is a small, configurable toolkit for exploring music "mood" clusters using numeric track features. It focuses on reproducible, configurable runs with tools for evaluation, snapshots, and comparisons so you can iterate on preprocessing and clustering settings and collect evidence for the best configuration.

Highlights
----------
- Config-driven batch runs (YAML or CLI) with separate outputs per run
- Preprocessing options: standard or min-max scaling, feature selection, PCA
- K selection (silhouette-based) and multi-metric run evaluation (silhouette, Davies–Bouldin, Calinski–Harabasz)
- Named snapshots (save/load scaler + clustering model + metadata)
- Run comparison (Adjusted Rand Index, NMI, overlap tables) and pairwise stability heatmaps
- CLI interactive UI for running experiments, comparing runs, building reports, and exporting evidence bundles
- Basic unit tests for preprocessing, snapshots and comparison utilities

Quick examples
--------------
Run clustering on a dataset (writes outputs to `outputs/`):

```bash
python main.py --input data/dataset_A.csv
```

Run a set of configurations from a YAML file:

```bash
python main.py --config configs/example_runs.yaml
```

Save a named snapshot (scaler+model+metadata) from a CLI run:

```bash
python main.py --input data/dataset_A.csv --output outputs/run_k3 --save-snapshot my_run_snapshot
```

Compare two runs from Python:

```python
from src.compare import compare_two_runs
res = compare_two_runs('outputs/run_k3', 'outputs/run_k4')
print(res['ari'], res['nmi'])
```

Generate a stability report across multiple runs:

```python
from src.stability import save_stability_report
save_stability_report(['outputs/run_k3', 'outputs/run_k4', 'outputs/run_minmax_3feat'])
```

Create a packaged zip report for a run:

```python
from src.reporting import build_zip_report
build_zip_report('outputs/run_k3')
```

Export an evidence bundle (copies selected run folders + run_history.json):

```python
from src.evidence import export_bundle
export_bundle(['outputs/run_k3', 'outputs/run_k4'], 'outputs/my_evidence_bundle')
```

Notes
-----
- For best alignment when comparing runs, include a stable identifier column (e.g. `id`, `track_id`) in `clustered_dataset.csv`. Otherwise comparisons align by row order.
- The project includes tests in `tests/` (run them directly as scripts or via pytest if installed).

Virtual environment (recommended)
--------------------------------
From the project root, create and activate a venv and install dependencies:

macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows (cmd)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

When finished, run `deactivate`.
