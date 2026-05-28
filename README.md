in cmd / terminal type
python main.py --input data/dataset_A.csv
or
python main.py --input data/dataset_B.csv

the output is visible in the outputs folder

Keywords: music, clustering, mood


For run based on config file, you can run this:
python3 main.py --config configs/example_runs.yaml

## Virtual environment (recommended)

From the project root, create and activate a project-local virtual environment and install dependencies:

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

Add `.venv/` to .gitignore and configure your editor to use the `.venv` interpreter. When finished, run `deactivate`.