"""Preprocess MedMCQA (`openlifescienceai/medmcqa`) for ATGen's
`task: multi-choice-qa` pipeline.

Source schema: {question, opa, opb, opc, opd, cop (int 0-3), subject_name, ...}
ATGen's multi-choice loader needs the four options bundled into a single
`choices` list and the answer as a letter ("A"/"B"/"C"/"D"). The MedMCQA test
split has no labels — we use `validation` as the eval split.

Run once:

    python -m atgen.utils.data.preprocess_medmcqa --out cache/medmcqa_preprocessed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


_LETTERS = "ABCD"


def build_row(row: dict) -> dict:
    cop = row.get("cop")
    if cop is None or cop < 0 or cop >= len(_LETTERS):
        answer = ""
    else:
        answer = _LETTERS[cop]
    return {
        "question": (row.get("question") or "").strip(),
        "choices": [
            (row.get("opa") or "").strip(),
            (row.get("opb") or "").strip(),
            (row.get("opc") or "").strip(),
            (row.get("opd") or "").strip(),
        ],
        "answer": answer,
        "subject": row.get("subject_name") or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="openlifescienceai/medmcqa",
        help="HF dataset id (default: openlifescienceai/medmcqa).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/medmcqa_preprocessed"),
        help="Output directory for the on-disk DatasetDict.",
    )
    args = parser.parse_args()

    ds = load_dataset(args.source)
    if "train" not in ds or "validation" not in ds:
        raise RuntimeError(
            f"Expected splits 'train' and 'validation' in {args.source}, "
            f"got {list(ds.keys())}"
        )

    out = DatasetDict()
    for src_split, dst_split in [("train", "train"), ("validation", "test")]:
        d = ds[src_split]
        original_cols = list(d.column_names)
        d = d.map(build_row, remove_columns=original_cols)
        d = d.filter(lambda r: bool(r["answer"]) and all(r["choices"]))
        out[dst_split] = d

    args.out.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(args.out))
    print(
        f"Wrote {len(out['train'])} train rows and {len(out['test'])} test rows "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
