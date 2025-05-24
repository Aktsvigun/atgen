from pathlib import Path
import json
import pandas as pd


# TODO: fix as it currently rewrites the file every time
def combine_results(workdir: str | Path, num_iterations: int) -> None:
    if isinstance(workdir, str):
        workdir = Path(workdir)
    with open(Path(workdir) / "iter_0" / "metrics.json") as f:
        metrics = [json.load(f)]
    for i in range(1, num_iterations + 1):
        with open(Path(workdir) / f"iter_{i}" / "metrics.json") as f:
            metrics += [json.load(f)]
    with open(workdir / "metrics.json", "w") as f:
        json.dump({i: val for i, val in enumerate(metrics)}, f)
    pd.DataFrame(metrics).to_csv(workdir / "metrics.csv", index=0)
