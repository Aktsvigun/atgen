"""Preprocess MMLU (`cais/mmlu`, subset `all`) for ATGen's
`task: multi-choice-qa` pipeline.

Source schema is {question, subject, choices (4 strings), answer (int 0-3)}.
ATGen's multi-choice loader (`_preprocess_multi_choice_qa`) expects the answer
column to be a *letter string* like "A"/"B"/"C"/"D"; this script does the
int → letter conversion and saves a DatasetDict with just a `test` split.

Run once:

    python -m atgen.utils.data.preprocess_mmlu --out cache/mmlu_preprocessed

Then point a data config (or `eval_dataset`) at the path:

    eval_dataset:
      dataset: 'cache/mmlu_preprocessed'
      test_split_name: test
      input_column_name: {"question": "question", "options": "choices"}
      output_column_name: 'answer'
      task: 'multi-choice-qa'
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_row(row: dict) -> dict:
    answer_idx = row.get("answer")
    if answer_idx is None or answer_idx < 0 or answer_idx >= len(_LETTERS):
        answer_letter = ""
    else:
        answer_letter = _LETTERS[answer_idx]
    return {
        "question": (row.get("question") or "").strip(),
        "choices": list(row.get("choices") or []),
        "answer": answer_letter,
        "subject": row.get("subject") or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="cais/mmlu",
        help="HF dataset id (default: cais/mmlu).",
    )
    parser.add_argument(
        "--subset",
        default="all",
        help="HF dataset subset (default: all).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which source split to take as the eval set (default: test).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/mmlu_preprocessed"),
        help="Output directory for the on-disk DatasetDict.",
    )
    args = parser.parse_args()

    d = load_dataset(args.source, args.subset, split=args.split)
    original_cols = list(d.column_names)
    d = d.map(build_row, remove_columns=original_cols)
    d = d.filter(lambda r: bool(r["answer"]) and len(r["choices"]) >= 2)

    out = DatasetDict({"test": d})
    args.out.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(args.out))
    print(f"Wrote {len(out['test'])} test rows to {args.out}")


if __name__ == "__main__":
    main()
