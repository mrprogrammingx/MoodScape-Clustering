import os
import shutil
from src.stability import save_stability_report


def make_fake_run(base, name, clusters):
    import pandas as pd
    p = os.path.join(base, name)
    os.makedirs(p, exist_ok=True)
    df = pd.DataFrame({"id": list(range(len(clusters))), "cluster": clusters})
    df.to_csv(os.path.join(p, "clustered_dataset.csv"), index=False)
    return p


def test_stability_small():
    base = "outputs/test_stability"
    if os.path.exists(base):
        shutil.rmtree(base)
    os.makedirs(base, exist_ok=True)

    r1 = make_fake_run(base, "r1", [0,0,1,1,2])
    r2 = make_fake_run(base, "r2", [0,0,1,2,2])
    r3 = make_fake_run(base, "r3", [1,1,0,0,2])

    report = save_stability_report([r1, r2, r3], out_folder=os.path.join(base, "out"))
    assert os.path.exists(report)

    shutil.rmtree(base)


if __name__ == "__main__":
    test_stability_small()
    print("stability test passed")
