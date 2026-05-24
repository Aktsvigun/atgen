"""Preprocess HotpotQA distractor for ATGen's `task: open-qa` pipeline.

`hotpotqa/hotpot_qa` (config `distractor`, ~90k train / 7k validation) ships
`context = {title: list[str], sentences: list[list[str]]}` — 10 paragraphs
(2 gold + 8 distractors). This script flattens that into a single prompt
string (same shape as MuSiQue's preprocessor) and keeps the gold `answer`.

  - `input`  : "Paragraphs:\n1. [title] sentence1 sentence2 ...\n2. ...\n\nQuestion: ..."
  - `answer` : gold short-span answer (string)

HotpotQA doesn't ship answer aliases, so `output_column_name` is just a string
(no need for an `all_answers` list as in MuSiQue).

Run once:

    python -m atgen.utils.data.preprocess_hotpotqa --out cache/hotpotqa_preprocessed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


def format_input(context: dict, question: str) -> str:
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    lines = ["Paragraphs:"]
    for i, (title, sents) in enumerate(zip(titles, sentences), 1):
        title = (title or "").strip()
        body = " ".join(s.strip() for s in (sents or []) if s and s.strip())
        prefix = f"[{title}] " if title else ""
        lines.append(f"{i}. {prefix}{body}")
    lines.append("")
    lines.append(f"Question: {question.strip()}")
    return "\n".join(lines)


def build_row(row: dict) -> dict:
    return {
        "input": format_input(row.get("context") or {}, row.get("question") or ""),
        "answer": (row.get("answer") or "").strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="hotpotqa/hotpot_qa",
        help="HF dataset id (default: hotpotqa/hotpot_qa).",
    )
    parser.add_argument(
        "--config",
        default="distractor",
        help="HF dataset config (default: distractor).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/hotpotqa_preprocessed"),
        help="Output directory for the on-disk DatasetDict.",
    )
    args = parser.parse_args()

    ds = load_dataset(args.source, args.config)
    if "train" not in ds or "validation" not in ds:
        raise RuntimeError(
            f"Expected splits 'train' and 'validation' in {args.source}/{args.config}, "
            f"got {list(ds.keys())}"
        )

    out = DatasetDict()
    for src_split, dst_split in [("train", "train"), ("validation", "test")]:
        d = ds[src_split]
        original_cols = list(d.column_names)
        d = d.map(build_row, remove_columns=original_cols)
        d = d.filter(lambda r: bool(r["answer"]))
        out[dst_split] = d

    args.out.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(args.out))
    print(
        f"Wrote {len(out['train'])} train rows and {len(out['test'])} test rows "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
