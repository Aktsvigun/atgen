"""Compare experiments side-by-side as a pivot table.

Rows = experiment_name (read from each run's resolved config.yaml).
Cols = query_size.
Cell = mean ± CI half-width (n=...) for the chosen metric, taken from the
*last* iteration in metrics.json (post-training eval).

Designed as the companion view for `run/investigate_random_curve.sh`, but
works for any set of runs that share an output_dir.

Examples:
    python -m atgen.utils.compare_experiments outputs/2026-05-02
    python -m atgen.utils.compare_experiments outputs/2026-05-02 \\
        --metric exact_match_math --md
    python -m atgen.utils.compare_experiments outputs/2026-05-02 \\
        --experiments rand_repl rand_sane rand_greedy
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .aggregate_metrics import (
    _load_config,
    _load_metrics,
    confidence_interval,
    find_run_dirs,
)


_PREFERRED_METRICS = (
    "exact_match_math", "exact_match", "rougeL", "rouge1", "bleu", "sacrebleu",
)


def load_run_for_compare(run_dir: Path) -> dict[str, Any]:
    cfg = _load_config(run_dir)
    metrics = _load_metrics(run_dir)
    return {
        "run_dir": str(run_dir),
        "experiment_name": str(OmegaConf.select(cfg, "experiment_name", default="?")),
        "strategy": OmegaConf.select(cfg, "al.strategy", default=None),
        "query_size": OmegaConf.select(cfg, "al.query_size", default=None),
        "seed": OmegaConf.select(cfg, "seed", default=None),
        "metrics": metrics,
    }


def _last_iter_metrics(metrics: dict) -> dict:
    """Return the metrics dict for the largest integer iter key.

    Falls back to the last inserted value if keys aren't ints.
    """
    if not metrics:
        return {}
    int_keyed = []
    for k in metrics.keys():
        try:
            int_keyed.append((int(k), k))
        except (TypeError, ValueError):
            pass
    if int_keyed:
        return metrics[max(int_keyed)[1]] or {}
    return list(metrics.values())[-1] or {}


def autodetect_metric(runs: list[dict]) -> str | None:
    """Pick a numeric metric present in every run; prefer well-known names."""
    sets: list[set[str]] = []
    for r in runs:
        m = _last_iter_metrics(r["metrics"])
        sets.append({k for k, v in m.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)})
    if not sets:
        return None
    common = set.intersection(*sets) if sets else set()
    for pref in _PREFERRED_METRICS:
        if pref in common:
            return pref
    return next(iter(sorted(common))) if common else None


def build_pivot(runs: list[dict], metric: str, confidence: float):
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in runs:
        v = _last_iter_metrics(r["metrics"]).get(metric)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            grouped[(r["experiment_name"], r["query_size"])].append(float(v))

    experiments = sorted({e for (e, _) in grouped})
    qsizes = sorted(
        {q for (_, q) in grouped if isinstance(q, (int, float))},
        key=lambda x: float(x),
    )
    table: dict[tuple, dict | None] = {}
    for exp in experiments:
        for q in qsizes:
            vals = grouped.get((exp, q))
            table[(exp, q)] = confidence_interval(vals, confidence) if vals else None
    return experiments, qsizes, table


def _fmt_cell(c: dict | None) -> str:
    if c is None:
        return "-"
    if c["half_width"] is None or c["n"] < 2:
        return f"{c['mean']:.3f} (n={c['n']})"
    return f"{c['mean']:.3f}±{c['half_width']:.3f} (n={c['n']})"


def format_pivot_text(experiments, qsizes, table, metric, confidence) -> str:
    pct = int(round(confidence * 100))
    name_w = max(12, max((len(e) for e in experiments), default=12)) + 2
    cells = [_fmt_cell(table[(e, q)]) for e in experiments for q in qsizes]
    col_w = max(12, max((len(c) for c in cells), default=12) + 2)

    lines = [
        f"Metric: {metric}    {pct}% CI    (cell: mean ± half-width (n))",
        f"{'experiment':<{name_w}}" + "".join(f"{'q=' + str(q):>{col_w}}" for q in qsizes),
    ]
    lines.append("-" * (name_w + col_w * len(qsizes)))
    for exp in experiments:
        row = f"{exp:<{name_w}}"
        for q in qsizes:
            row += f"{_fmt_cell(table[(exp, q)]):>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


def format_pivot_markdown(experiments, qsizes, table, metric, confidence) -> str:
    pct = int(round(confidence * 100))
    lines = [f"**{metric}** — mean ± {pct}% CI half-width (n)"]
    lines.append("| experiment | " + " | ".join(f"q={q}" for q in qsizes) + " |")
    lines.append("|" + "---|" * (len(qsizes) + 1))
    for exp in experiments:
        row = ["`" + exp + "`"]
        for q in qsizes:
            row.append(_fmt_cell(table[(exp, q)]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("roots", nargs="+", type=Path,
                   help="One or more results root dirs (e.g. outputs/2026-05-02).")
    p.add_argument("--metric", default=None,
                   help="Metric name to pivot. Auto-detected if omitted.")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="Confidence level (default: 0.95).")
    p.add_argument("--experiments", nargs="*", default=None,
                   help="Restrict to these experiment_name values.")
    p.add_argument("--md", action="store_true", help="Markdown table output.")
    p.add_argument("--csv", type=Path, default=None,
                   help="Optional CSV output (long format).")
    args = p.parse_args()

    if not 0 < args.confidence < 1:
        p.error("--confidence must be in (0, 1)")

    runs: list[dict] = []
    failed: list[tuple[Path, str]] = []
    for run_dir in find_run_dirs(args.roots):
        try:
            runs.append(load_run_for_compare(run_dir))
        except Exception as e:
            failed.append((run_dir, str(e)))

    if args.experiments:
        keep = set(args.experiments)
        runs = [r for r in runs if r["experiment_name"] in keep]

    if failed:
        print(f"Skipped {len(failed)} run(s) that could not be loaded.")
    if not runs:
        print("No runs to compare.")
        return

    metric = args.metric or autodetect_metric(runs)
    if metric is None:
        print("Could not auto-detect a metric. Pass --metric explicitly.")
        return

    experiments, qsizes, table = build_pivot(runs, metric, args.confidence)
    print(f"Loaded {len(runs)} run(s); {len(experiments)} experiment(s); "
          f"{len(qsizes)} query size(s).")
    print()
    if args.md:
        print(format_pivot_markdown(experiments, qsizes, table, metric, args.confidence))
    else:
        print(format_pivot_text(experiments, qsizes, table, metric, args.confidence))

    if args.csv:
        rows = ["experiment,query_size,n,mean,std,sem,ci_low,ci_high,half_width"]
        for exp in experiments:
            for q in qsizes:
                c = table[(exp, q)]
                if c is None:
                    rows.append(f"{exp},{q},0,,,,,,")
                else:
                    rows.append(",".join([
                        exp, str(q), str(c["n"]),
                        *("" if c[k] is None else f"{c[k]}"
                          for k in ("mean", "std", "sem", "ci_low", "ci_high", "half_width")),
                    ]))
        args.csv.write_text("\n".join(rows) + "\n")
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
