"""Preprocess MuSiQue for ATGen's `task: open-qa` pipeline.

MuSiQue (`dgslibisey/MuSiQue`) ships paragraphs as a list of dicts and answers
with an aliases list — neither maps cleanly onto ATGen's flat-column expectation
in `get_preprocess_function` / `_preprocess_multicolumn_labels_if_needed`. This
script does a one-shot flatten and saves a HuggingFace DatasetDict to disk:

    train split columns: {id, input, answer, all_answers}
    test  split columns: {id, input, answer, all_answers}

  - `input`        : "Paragraphs:\n1. [title] text\n2. ...\n\nQuestion: ..."
  - `answer`       : gold answer (used as training target)
  - `all_answers`  : [answer, *aliases] (used as test reference, so the
                     open-qa exact_match scorer matches any).

Run once:

    python -m atgen.utils.data.preprocess_musique --out cache/musique_preprocessed

Then point a data config at the path:

    dataset: 'cache/musique_preprocessed'
    output_column_name: {'train': 'answer', 'test': 'all_answers'}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


def format_input(paragraphs: list[dict], question: str) -> str:
    lines = ["Paragraphs:"]
    for i, p in enumerate(paragraphs, 1):
        title = (p.get("title") or "").strip()
        text = (p.get("paragraph_text") or "").strip()
        prefix = f"[{title}] " if title else ""
        lines.append(f"{i}. {prefix}{text}")
    lines.append("")
    lines.append(f"Question: {question.strip()}")
    return "\n".join(lines)


def build_row(row: dict) -> dict:
    answer = (row.get("answer") or "").strip()
    aliases = [a.strip() for a in (row.get("answer_aliases") or []) if a and a.strip()]
    all_answers = [answer] + [a for a in aliases if a != answer]
    return {
        "input": format_input(row["paragraphs"], row["question"]),
        "answer": answer,
        "all_answers": all_answers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="dgslibisey/MuSiQue",
                        help="HF dataset id (default: dgslibisey/MuSiQue).")
    parser.add_argument("--out", type=Path, default=Path("cache/musique_preprocessed"),
                        help="Output directory for the on-disk DatasetDict.")
    parser.add_argument("--keep-unanswerable", action="store_true",
                        help="Don't drop rows where `answerable` is False.")
    args = parser.parse_args()

    ds = load_dataset(args.source)
    if "train" not in ds or "validation" not in ds:
        raise RuntimeError(f"Expected splits 'train' and 'validation' in {args.source}, "
                           f"got {list(ds.keys())}")

    out = DatasetDict()
    for src_split, dst_split in [("train", "train"), ("validation", "test")]:
        d = ds[src_split]
        if not args.keep_unanswerable and "answerable" in d.column_names:
            d = d.filter(lambda r: bool(r["answerable"]))
        original_cols = list(d.column_names)
        d = d.map(build_row, remove_columns=original_cols)
        out[dst_split] = d

    args.out.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(args.out))
    print(f"Wrote {len(out['train'])} train rows and {len(out['test'])} test rows to {args.out}")


if __name__ == "__main__":
    main()
