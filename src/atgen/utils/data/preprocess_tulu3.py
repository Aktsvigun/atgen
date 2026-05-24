"""Preprocess Tulu-3 SFT-mixture as a flat (input, answer) training pool for
ATGen's `task: open-qa`-style pipeline.

`allenai/tulu-3-sft-mixture` (~939k rows) ships multi-turn `messages` columns.
For AL purposes we collapse to single-turn: the first user message becomes
`input`, the first assistant response becomes `answer`. Rows that don't have
both are dropped. Optional `--max-rows N` caps the pool (useful for pilots).

The output is a DatasetDict with a single `train` split, so this dataset can be
pointed at via `data.dataset` in a YAML config. Pair it with `eval_dataset:` on
a different benchmark (e.g. cache/mmlu_preprocessed) to do instruct-tuning AL
in the DEITA/MIG-style setup.

Run once:

    python -m atgen.utils.data.preprocess_tulu3 --out cache/tulu3_preprocessed
    # or, for a smaller pilot:
    python -m atgen.utils.data.preprocess_tulu3 --out cache/tulu3_50k --max-rows 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


def flatten_messages(messages: list[dict]) -> dict[str, str] | None:
    """Take the first user message and the first assistant message that follows
    it. Returns None if either is missing or empty."""
    user_text = None
    assistant_text = None
    for msg in messages or []:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user" and user_text is None:
            user_text = content
        elif role == "assistant" and user_text is not None and assistant_text is None:
            assistant_text = content
            break
    if not user_text or not assistant_text:
        return None
    return {"input": user_text, "answer": assistant_text}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="allenai/tulu-3-sft-mixture",
        help="HF dataset id (default: allenai/tulu-3-sft-mixture).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/tulu3_preprocessed"),
        help="Output directory for the on-disk DatasetDict.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap the train pool at this many rows (shuffled). None = keep all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed when --max-rows is set.",
    )
    args = parser.parse_args()

    ds = load_dataset(args.source, split="train")
    rows: list[dict[str, str]] = []
    for row in ds:
        flat = flatten_messages(row.get("messages") or [])
        if flat is None:
            continue
        rows.append(flat)

    out_ds = Dataset.from_list(rows)
    if args.max_rows is not None and len(out_ds) > args.max_rows:
        out_ds = out_ds.shuffle(seed=args.seed).select(range(args.max_rows))

    out = DatasetDict({"train": out_ds})
    args.out.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(args.out))
    print(
        f"Wrote {len(out['train'])} train rows to {args.out} "
        f"(dropped {len(ds) - len(rows)} rows with no usable user+assistant turn)"
    )


if __name__ == "__main__":
    main()
