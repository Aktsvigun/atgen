"""Aggregate metrics across active-learning experiment runs.

Walks one or more results directories, groups runs by (strategy, query_size),
and reports mean +/- confidence interval per iteration / per metric.

Each run is expected to contain `config.yaml` and `metrics.json` (the format
produced by `combine_results`: ``{"<iter>": {"<metric>": value, ...}, ...}``).

Examples:
    python -m atgen.utils.aggregate_metrics outputs/2026-04-27
    python -m atgen.utils.aggregate_metrics outputs/2026-04-26 outputs/2026-04-27 \\
        --confidence 0.95 --out aggregated.json --csv aggregated.csv
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf


def _has_metrics(run_dir: Path) -> bool:
    if (run_dir / "metrics.json").exists():
        return True
    return any(run_dir.glob("iter_*/metrics.json"))


def find_run_dirs(roots: Iterable[Path]) -> Iterable[Path]:
    """Yield experiment run directories under the given roots.

    A run dir is any directory with `config.yaml` and either a top-level
    `metrics.json` (the rollup written by `combine_results`) or at least one
    `iter_*/metrics.json` (so partially-finished runs are still included).
    """
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"warning: {root} does not exist, skipping")
            continue
        if (root / "config.yaml").exists() and _has_metrics(root):
            yield root
            continue
        for cfg in root.rglob("config.yaml"):
            run_dir = cfg.parent
            if run_dir.name.startswith("iter_"):
                continue
            if _has_metrics(run_dir):
                yield run_dir


def _load_metrics(run_dir: Path) -> dict:
    """Return the per-iter metrics dict, falling back to iter_*/metrics.json."""
    top = run_dir / "metrics.json"
    if top.exists():
        with top.open() as f:
            return json.load(f)
    out: dict[str, dict] = {}
    for iter_metrics in sorted(run_dir.glob("iter_*/metrics.json")):
        # iter_dir name is "iter_<n>"; key is "<n>"
        iter_key = iter_metrics.parent.name.split("_", 1)[1]
        with iter_metrics.open() as f:
            out[iter_key] = json.load(f)
    return out


def _config_from_overrides(overrides_path: Path):
    """Reconstruct the minimal grouping fields from `.hydra/overrides.yaml`.

    overrides.yaml is a flat list like ["al=random", "al.query_size=20",
    "experiment_name=rand_repl", ...]. Enough to recover (strategy,
    query_size, experiment_name, seed) when both resolved configs are corrupt.
    """
    items: list[str] = []
    with overrides_path.open() as f:
        for line in f:
            line = line.strip()
            if not line.startswith("- "):
                continue
            items.append(line[2:].strip())
    cfg: dict[str, Any] = {"al": {}}
    for raw in items:
        s = raw.lstrip("+").lstrip("~")
        if "=" not in s:
            continue
        key, val = s.split("=", 1)
        # cast obvious numerics
        cast: Any = val
        try:
            cast = int(val)
        except ValueError:
            try:
                cast = float(val)
            except ValueError:
                pass
        # set nested dotted key. Single-key overrides like `al=random` are
        # Hydra config-group selectors, not value assignments: skip if they
        # would clobber a dict (e.g. cfg["al"] already holds query_size).
        parts = key.split(".")
        d = cfg
        try:
            for p in parts[:-1]:
                nxt = d.get(p)
                if not isinstance(nxt, dict):
                    nxt = {}
                    d[p] = nxt
                d = nxt
            existing = d.get(parts[-1])
            if isinstance(existing, dict):
                continue
            d[parts[-1]] = cast
        except AttributeError:
            continue
    return OmegaConf.create(cfg)


def _load_config(run_dir: Path):
    """Load resolved config with progressively cruder fallbacks.

    Order: outer `config.yaml` (full resolved) → `.hydra/config.yaml` (Hydra's
    snapshot) → `.hydra/overrides.yaml` (just the CLI overrides). Runs killed
    mid-write can leave either of the first two truncated; the overrides file
    is written once at startup and is the most reliable source.
    """
    candidates = [
        run_dir / "config.yaml",
        run_dir / ".hydra" / "config.yaml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return OmegaConf.load(path)
        except Exception:
            continue
    overrides = run_dir / ".hydra" / "overrides.yaml"
    if overrides.exists():
        try:
            return _config_from_overrides(overrides)
        except Exception:
            pass
    raise FileNotFoundError(f"no readable config in {run_dir}")


def load_run(run_dir: Path) -> dict[str, Any]:
    cfg = _load_config(run_dir)
    metrics = _load_metrics(run_dir)

    al = OmegaConf.select(cfg, "al", default={}) or {}
    return {
        "run_dir": str(run_dir),
        "strategy": OmegaConf.select(al, "strategy", default=None),
        "query_size": OmegaConf.select(al, "query_size", default=None),
        "metrics": metrics,
    }


def _t_critical(n: int, confidence: float) -> float:
    """Two-tailed t critical value with a normal-approx fallback."""
    if n < 2:
        return float("nan")
    alpha = 1.0 - confidence
    try:
        from scipy.stats import t

        return float(t.ppf(1 - alpha / 2, df=n - 1))
    except ImportError:
        from statistics import NormalDist

        return NormalDist().inv_cdf(1 - alpha / 2)


def confidence_interval(values: list[float], confidence: float = 0.95) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "sem": None,
                "ci_low": None, "ci_high": None, "half_width": None}
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "std": None, "sem": None,
                "ci_low": mean, "ci_high": mean, "half_width": 0.0}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    half_width = _t_critical(n, confidence) * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "half_width": half_width,
    }


def aggregate(runs: list[dict[str, Any]], confidence: float = 0.95) -> dict[tuple, dict]:
    """Group runs by (strategy, query_size) and compute CIs per iter / per metric."""
    grouped: dict[tuple, list] = defaultdict(list)
    for r in runs:
        grouped[(r["strategy"], r["query_size"])].append(r)

    out: dict[tuple, dict] = {}
    for key, group_runs in grouped.items():
        per_iter_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in group_runs:
            for it_key, it_metrics in (r["metrics"] or {}).items():
                if not isinstance(it_metrics, dict):
                    continue
                for metric_name, value in it_metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        per_iter_metric[str(it_key)][metric_name].append(float(value))

        def _iter_sort_key(k: str):
            try:
                return (0, int(k))
            except ValueError:
                return (1, k)

        iters_out: dict[str, dict[str, dict]] = {}
        for it_key in sorted(per_iter_metric.keys(), key=_iter_sort_key):
            iters_out[it_key] = {
                m: confidence_interval(vals, confidence)
                for m, vals in sorted(per_iter_metric[it_key].items())
            }

        out[key] = {
            "n_runs": len(group_runs),
            "run_dirs": [r["run_dir"] for r in group_runs],
            "iterations": iters_out,
        }
    return out


def format_table(aggregated: dict[tuple, dict], confidence: float) -> str:
    pct = round(confidence * 100, 1)
    pct_label = f"CI{int(pct) if pct == int(pct) else pct}%"
    lines: list[str] = []
    sorted_keys = sorted(
        aggregated.keys(),
        key=lambda k: (str(k[0]) if k[0] is not None else "", k[1] if isinstance(k[1], (int, float)) else 0),
    )
    for (strategy, qs) in sorted_keys:
        data = aggregated[(strategy, qs)]
        lines.append(
            f"\n=== strategy={strategy}  query_size={qs}  n_runs={data['n_runs']} ==="
        )
        lines.append(
            f"{'iter':<6}{'metric':<32}{'n':>4}  {'mean':>10}  {'std':>10}  {pct_label:>26}"
        )
        lines.append("-" * 92)
        for it_key, metrics in data["iterations"].items():
            for metric, s in metrics.items():
                mean_s = "n/a" if s["mean"] is None else f"{s['mean']:.4f}"
                std_s = "n/a" if s["std"] is None else f"{s['std']:.4f}"
                if s["ci_low"] is None:
                    ci_s = "n/a"
                else:
                    ci_s = f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}]"
                lines.append(
                    f"{it_key:<6}{metric:<32}{s['n']:>4}  {mean_s:>10}  {std_s:>10}  {ci_s:>26}"
                )
    return "\n".join(lines)


def aggregated_to_serializable(aggregated: dict[tuple, dict]) -> list[dict]:
    rows = []
    for (strategy, qs), data in aggregated.items():
        rows.append({
            "strategy": strategy,
            "query_size": qs,
            "n_runs": data["n_runs"],
            "run_dirs": data["run_dirs"],
            "iterations": data["iterations"],
        })
    return rows


def write_csv(aggregated: dict[tuple, dict], path: Path) -> None:
    import csv

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "query_size", "iter", "metric",
                    "n", "mean", "std", "sem", "ci_low", "ci_high", "half_width"])
        for (strategy, qs), data in aggregated.items():
            for it_key, metrics in data["iterations"].items():
                for metric, s in metrics.items():
                    w.writerow([
                        strategy, qs, it_key, metric,
                        s["n"], s["mean"], s["std"], s["sem"],
                        s["ci_low"], s["ci_high"], s["half_width"],
                    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("roots", nargs="+", type=Path,
                        help="One or more directories with results (e.g. outputs/2026-04-27).")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for the interval (default: 0.95).")
    parser.add_argument("--out", type=Path, default='results/aggregated.json',
                        help="Optional path to write aggregated results as JSON.")
    parser.add_argument("--csv", type=Path, default='results/aggregated.csv',
                        help="Optional path to write aggregated results as CSV (long format).")
    args = parser.parse_args()

    if not 0 < args.confidence < 1:
        parser.error("--confidence must be in (0, 1)")

    runs: list[dict[str, Any]] = []
    failed: list[tuple[Path, str]] = []
    for run_dir in find_run_dirs(args.roots):
        try:
            runs.append(load_run(run_dir))
        except Exception as e:
            failed.append((run_dir, str(e)))

    if failed:
        print(f"Skipped {len(failed)} run(s) that could not be loaded:")
        for d, err in failed[:10]:
            print(f"  - {d}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    if not runs:
        print("No runs found.")
        return

    print(f"Loaded {len(runs)} run(s) from {len(args.roots)} root(s).")
    aggregated = aggregate(runs, confidence=args.confidence)
    print(format_table(aggregated, args.confidence))

    if args.out:
        with args.out.open("w") as f:
            json.dump(aggregated_to_serializable(aggregated), f, indent=2)
        print(f"\nWrote JSON: {args.out}")
    if args.csv:
        write_csv(aggregated, args.csv)
        print(f"Wrote CSV:  {args.csv}")


if __name__ == "__main__":
    main()
