#!/usr/bin/env python
"""Make the gpt-4o-mini draft labels easy to skim and correct.

Two modes:

  EXPORT (default) — write a spreadsheet-friendly review file, sorted toxic-first
  (toxic is where the bootstrap is least reliable, so review it first):
        uv run python scripts/review_labels.py
    -> data/labels_review.csv  with columns: id, label, fix, note, text, permalink
       Open it in Excel/Sheets. To change a label, type the correct one in the `fix`
       column (constructive | neutral | toxic). Leave `fix` blank to keep `label`.

  APPLY — read your edits back into labeled.jsonl, then you rebuild the CSV:
        uv run python scripts/review_labels.py --apply
        uv run python scripts/build_csv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

LABELED = Path("data/labeled.jsonl")
REVIEW = Path("data/labels_review.csv")
VALID = {"constructive", "neutral", "toxic"}
ORDER = {"toxic": 0, "neutral": 1, "constructive": 2}  # review toxic first


def export() -> None:
    df = pd.read_json(LABELED, lines=True)
    df["fix"] = ""
    df["_o"] = df["label"].map(ORDER).fillna(9)
    df = df.sort_values(["_o", "id"])
    cols = ["id", "label", "fix", "note", "text", "permalink"]
    df[cols].to_csv(REVIEW, index=False)
    n = df["label"].value_counts()
    print(f"  wrote {len(df)} rows -> {REVIEW}")
    print("  by label:", {k: int(n.get(k, 0)) for k in ["toxic", "neutral", "constructive"]})
    print("  Open it, fill the `fix` column only where the label is wrong, save, then:")
    print("    uv run python scripts/review_labels.py --apply && uv run python scripts/build_csv.py")


def apply() -> None:
    if not REVIEW.exists():
        sys.exit(f"  ERROR: {REVIEW} not found — run export mode first.")
    review = pd.read_csv(REVIEW).set_index("id")
    labeled = pd.read_json(LABELED, lines=True)

    n_fixed = 0
    for i, row in labeled.iterrows():
        rid = row["id"]
        if rid in review.index:
            fix = str(review.loc[rid, "fix"]).strip().lower()
            if fix in VALID and fix != row["label"]:
                labeled.at[i, "label"] = fix
                labeled.at[i, "note"] = "(human-corrected)"
                labeled.at[i, "reviewed"] = True
                n_fixed += 1
            elif fix and fix not in VALID:
                print(f"  warn: '{fix}' (id {rid}) is not a valid label — skipped")
    # mark everything that appeared in the review file as reviewed
    labeled["reviewed"] = labeled["id"].isin(review.index)
    labeled.to_json(LABELED, orient="records", lines=True, force_ascii=False)
    print(f"  applied {n_fixed} corrections -> {LABELED}")
    print("  next: uv run python scripts/build_csv.py")


if __name__ == "__main__":
    apply() if "--apply" in sys.argv else export()
