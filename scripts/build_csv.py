#!/usr/bin/env python
"""Build the notebook-ready CSV from labeled.jsonl.

    data/labeled.jsonl  ->  data/valorant_takes.csv   (columns: text,label)

Does the heavy text cleaning (so raw stays faithful), validates labels against the
LABEL_MAP, dedupes, and prints the label distribution so you can see if any class is
under the 20% target before you waste a Colab run.

RUN:
    uv run python scripts/build_csv.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

LABELS = {"constructive", "neutral", "toxic"}  # must match notebook LABEL_MAP keys
IN = Path("data/labeled.jsonl")
OUT = Path("data/valorant_takes.csv")
# Downsample the majority classes to this cap so every class clears the ≥20% target
# (toxic is the natural minority on a moderated subreddit). Minority class is kept in
# full if it's under the cap. Set to None to disable balancing. Seeded for repeatability.
BALANCE_CAP = 130
BALANCE_SEED = 42

# ── cleaning ──────────────────────────────────────────────────────────────
URL_RE = re.compile(r"https?://\S+")
QUOTE_LINE_RE = re.compile(r"^\s*>.*$", re.MULTILINE)   # markdown quote blocks
MENTION_RE = re.compile(r"(?<!\w)/?u/[A-Za-z0-9_-]+|@[A-Za-z0-9_-]+")
SUBREDDIT_RE = re.compile(r"(?<!\w)/?r/[A-Za-z0-9_]+")
MD_ARTIFACT_RE = re.compile(r"[*_`#]+")                 # bold/italic/code/heading marks
WS_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    text = QUOTE_LINE_RE.sub(" ", text)   # drop quoted parent text -> not the author's voice
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = SUBREDDIT_RE.sub(" ", text)
    text = MD_ARTIFACT_RE.sub("", text)
    text = text.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    text = WS_RE.sub(" ", text).strip()
    return text


def main() -> None:
    if not IN.exists():
        sys.exit(f"  ERROR: {IN} not found. Label data first (see scripts/label_bootstrap.py "
                 "or hand-label), writing one JSON object per line with at least 'text' and 'label'.")
    df = pd.read_json(IN, lines=True)

    if "label" not in df or "text" not in df:
        sys.exit("  ERROR: labeled.jsonl needs 'text' and 'label' fields.")

    df["text"] = df["text"].astype(str).map(clean)
    df = df[df["text"].str.len() >= 10]                 # drop rows that cleaned to ~nothing
    df = df.drop_duplicates(subset="text")

    bad = set(df["label"].unique()) - LABELS
    if bad:
        sys.exit(f"  ERROR: labels not in LABEL_MAP: {bad}. Fix these in labeled.jsonl.")

    df = df[["text", "label"]]

    if BALANCE_CAP:
        before = len(df)
        # iterate groups (keeps the label column, unlike groupby.apply in pandas 3.0)
        parts = [g.sample(min(len(g), BALANCE_CAP), random_state=BALANCE_SEED)
                 for _, g in df.groupby("label")]
        df = pd.concat(parts, ignore_index=True)
        print(f"  balanced: capped each class at {BALANCE_CAP} ({before} -> {len(df)} rows)\n")

    df = df.sample(frac=1, random_state=BALANCE_SEED).reset_index(drop=True)  # shuffle
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # pandas handles commas/quotes/newlines in text via proper CSV quoting
    df.to_csv(OUT, index=False)

    print(f"  wrote {len(df)} rows -> {OUT}\n")
    counts = df["label"].value_counts()
    print("  label distribution:")
    for label in sorted(LABELS):
        n = int(counts.get(label, 0))
        pct = 100 * n / len(df) if len(df) else 0
        flag = "  <-- under 20%!" if pct < 20 else ""
        print(f"    {label:<13} {n:>4}  ({pct:4.1f}%){flag}")
    if len(df) < 200:
        print(f"\n  NOTE: only {len(df)} rows — task requires >=200. Label more.")


if __name__ == "__main__":
    main()
