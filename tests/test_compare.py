import os
import shutil
import pandas as pd
from src.compare import compare_two_runs


def make_run_folder(base, name, clusters):
    folder = os.path.join(base, name)
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame({"id": list(range(len(clusters))), "cluster": clusters})
    df.to_csv(os.path.join(folder, "clustered_dataset.csv"), index=False)
    return folder


def test_compare_simple():
    base = "outputs/test_compare"
    if os.path.exists(base):
        shutil.rmtree(base)
    os.makedirs(base, exist_ok=True)

    # create two simple clusterings
    run1 = make_run_folder(base, "run1", [0, 0, 1, 1, 2, 2])
    run2 = make_run_folder(base, "run2", [0, 0, 1, 1, 2, 2])
    res_same = compare_two_runs(run1, run2)
    assert res_same["ari"] == 1.0

    # slightly different
    run3 = make_run_folder(base, "run3", [0, 1, 1, 1, 2, 2])
    res_diff = compare_two_runs(run1, run3)
    assert 0.0 <= res_diff["ari"] <= 1.0

    shutil.rmtree(base)


if __name__ == "__main__":
    test_compare_simple()
    print("compare tests passed")
