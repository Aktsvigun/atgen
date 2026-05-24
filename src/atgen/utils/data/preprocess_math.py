"""Preprocess Hendrycks competition_math for ATGen's `task: math` pipeline.

Source ships {problem, level, type, solution}. The final answer is buried
inside the LaTeX solution as `\\boxed{...}`. The original `hendrycks/competition_math`
HF repo was deauthorized; this script defaults to the `lighteval/MATH` mirror
(same 7.5k train / 5k test, identical schema). Pass `--source` to use a
different mirror (`qwedsacf/competition_math`, `EleutherAI/hendrycks_math`, …).

This script:

  - extracts the boxed answer into a separate `answer` column (test reference),
  - keeps the full LaTeX `solution` as the training target (`short_solution`),
  - filters rows where no `\\boxed{}` can be extracted (a handful exist),
  - saves a HuggingFace DatasetDict to disk.

Run once:

    python -m atgen.utils.data.preprocess_math --out cache/math_hendrycks_preprocessed

Then point a data config at the path:

    dataset: 'cache/math_hendrycks_preprocessed'
    output_column_name: {'train': 'short_solution', 'test': 'answer'}
    task: 'math'
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import DatasetDict, load_dataset


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(solution: str) -> str | None:
    matches = _BOXED_RE.findall(solution or "")
    return matches[-1].strip() if matches else None


def build_row(row: dict) -> dict:
    sol = row.get("solution") or ""
    answer = extract_boxed(sol) or ""
    return {
        "problem": (row.get("problem") or "").strip(),
        "short_solution": sol.strip(),
        "answer": answer,
        "level": row.get("level") or "",
        "type": row.get("type") or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="lighteval/MATH",
        help=(
            "HF dataset id (default: lighteval/MATH — the canonical mirror "
            "since `hendrycks/competition_math` was deauthorized)."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional HF dataset config name. lighteval/MATH has an `all` "
            "config that covers every subject — pass --config all if the "
            "default config fails to load."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/math_hendrycks_preprocessed"),
        help="Output directory for the on-disk DatasetDict.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=5000,
        help=(
            "When the source ships only a `train` split, hold out this many "
            "rows for the test set (default: 5000, matching the canonical "
            "Hendrycks split size)."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Shuffle seed for the auto-split (default: 42).",
    )
    args = parser.parse_args()

    if args.config:
        ds = load_dataset(args.source, args.config)
    else:
        ds = load_dataset(args.source)

    if "train" in ds and "test" in ds:
        train_raw, test_raw = ds["train"], ds["test"]
    elif "train" in ds:
        # Some mirrors (e.g. qwedsacf/competition_math) ship the full 12.5k as
        # one `train` split. Hold out args.test_size rows for a deterministic
        # AL eval set. This won't match the canonical Hendrycks test split row-
        # for-row, but it's a clean held-out set, which is all AL needs.
        full = ds["train"]
        if len(full) <= args.test_size:
            raise RuntimeError(
                f"{args.source} has only {len(full)} rows in `train`, can't "
                f"hold out {args.test_size} for test. Pass --test-size N."
            )
        full = full.shuffle(seed=args.split_seed)
        test_raw = full.select(range(args.test_size))
        train_raw = full.select(range(args.test_size, len(full)))
        print(
            f"{args.source} ships only `train`; auto-split into "
            f"{len(train_raw)} train + {len(test_raw)} test "
            f"(seed={args.split_seed})."
        )
    else:
        raise RuntimeError(
            f"Expected at least a `train` split in {args.source}, "
            f"got {list(ds.keys())}"
        )

    out = DatasetDict()
    for split, raw in [("train", train_raw), ("test", test_raw)]:
        original_cols = list(raw.column_names)
        d = raw.map(build_row, remove_columns=original_cols)
        d = d.filter(lambda r: bool(r["answer"]))
        out[split] = d

    args.out.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(args.out))
    print(
        f"Wrote {len(out['train'])} train rows and {len(out['test'])} test rows "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
