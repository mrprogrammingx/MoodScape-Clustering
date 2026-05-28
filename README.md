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