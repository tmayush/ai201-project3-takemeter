#!/usr/bin/env python
"""Bootstrap labels so you REVIEW instead of label-from-scratch.

    data/raw/comments.jsonl  ->  data/labeled.jsonl   (+ label, + note)

Uses **Azure OpenAI gpt-4o-mini** by default (falls back to Groq if Azure creds are
absent). Using gpt-4o-mini here is deliberate: the required baseline (notebook
Section 5) is Groq `llama-3.3-70b`, so the *label source* must be a different model —
otherwise you'd be training DistilBERT to imitate the baseline and grading the
baseline against its own guesses. gpt-4o-mini for labels, llama for the baseline, and
your human review on top = an honest comparison.

This is still a *draft*. Open labeled.jsonl and fix what's wrong (task #4) — that
judgment is the graded skill, and it surfaces your 3 hard-to-label cases for free.

ENV (.env at repo root):
    AZURE_FOUNDRY_ENDPOINT=https://....services.ai.azure.com/openai/v1
    AZURE_FOUNDRY_API_KEY=...
    AZURE_FOUNDRY_DEPLOYMENT=gpt-4o-mini
    # or, fallback:  GROQ_API_KEY=...

RUN:
    uv run python scripts/label_bootstrap.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

IN = Path("data/raw/comments.jsonl")
OUT = Path("data/labeled.jsonl")
VALID = {"constructive", "neutral", "toxic"}

# Mirrors planning.md. Keep in sync with that file and the notebook Groq prompt.
SYSTEM_PROMPT = """You label Valorant community comments (Reddit) by DISCOURSE HEALTH \
— how the comment contributes to the conversation — NOT whether its opinion is correct.

Assign exactly one label:

constructive — adds value: reasoned argument, helpful advice (lineups, strats,
  settings), thoughtful analysis, or a respectful answer, even when disagreeing.
neutral — on-topic but low-substance and non-hostile: casual chatter, jokes, simple
  reactions, one-line facts, plain questions, undirected venting ("ranked is
  miserable"), pure noise / off-topic / emoji-only.
toxic — hostile or corrosive: insults, flaming, personal attacks, teammate-blaming as
  abuse, slurs, contemptuous dismissal, sarcastic put-downs aimed at a person.

Edge rules: helpful point in a rude wrapper -> toxic only if the insult dominates,
else constructive. Respectful disagreement with reasoning -> constructive. Undirected
frustration with no target -> neutral (unless it contains slurs).

Respond with ONLY a compact JSON object: {"label":"<one label>","note":"<<=8 word reason>"}"""


# ── provider plumbing ─────────────────────────────────────────────────────
def make_caller():
    ep = os.environ.get("AZURE_FOUNDRY_ENDPOINT", "").rstrip("/")
    az_key = os.environ.get("AZURE_FOUNDRY_API_KEY")
    dep = os.environ.get("AZURE_FOUNDRY_DEPLOYMENT", "gpt-4o-mini")
    if ep and az_key:
        url = ep + "/chat/completions"
        headers = {"Authorization": f"Bearer {az_key}", "Content-Type": "application/json"}
        model = dep
        print(f"  labeler: Azure OpenAI ({dep})")
    elif os.environ.get("GROQ_API_KEY"):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                   "Content-Type": "application/json"}
        model = "llama-3.1-8b-instant"  # NOT the 70b baseline model
        print("  labeler: Groq (llama-3.1-8b-instant)")
    else:
        sys.exit("  ERROR: set AZURE_FOUNDRY_* or GROQ_API_KEY in .env")

    session = requests.Session()

    def call(text: str) -> tuple[str, str]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Comment:\n{text}"},
            ],
            "temperature": 0,
            "max_tokens": 60,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(5):
            r = session.post(url, headers=headers, json=payload, timeout=40)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            # Azure content filter rejects some inputs with 400 — that the comment
            # tripped a hostility/profanity filter is itself a strong `toxic` signal,
            # so default it to toxic-for-review instead of crashing the whole run.
            if r.status_code == 400:
                body = r.text.lower()
                if "content_filter" in body or "content management" in body or "filtered" in body:
                    return "toxic", "(azure content-filtered -> likely toxic; REVIEW)"
                return "neutral", "(400 bad request -> defaulted; REVIEW)"
            if 500 <= r.status_code < 600:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            try:
                obj = json.loads(raw)
                label = str(obj.get("label", "")).lower().strip()
                note = str(obj.get("note", ""))[:80]
            except Exception:
                label, note = "", ""
            if label not in VALID:
                return "neutral", "(unparseable -> defaulted; REVIEW)"
            return label, note
        return "neutral", "(rate-limited -> defaulted; REVIEW)"

    return call


def main() -> None:
    if not IN.exists():
        sys.exit(f"  ERROR: {IN} not found — run scripts/scrape_arctic.py first.")
    rows = [json.loads(l) for l in IN.read_text(encoding="utf-8").splitlines() if l.strip()]

    done: set[str] = set()
    if OUT.exists():  # resume
        done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}

    call = make_caller()
    n_new = 0
    with OUT.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            label, note = call(row["text"])
            f.write(json.dumps({
                "id": row["id"],
                "text": row["text"],
                "label": label,
                "note": note,
                "permalink": row.get("permalink", ""),
                "reviewed": False,
            }, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if i % 50 == 0:
                print(f"  {i}/{len(rows)} ...")
            time.sleep(0.15)

    print(f"\n  labeled {n_new} new comments -> {OUT}")
    print("  NEXT: review labeled.jsonl, then: uv run python scripts/build_csv.py")


if __name__ == "__main__":
    main()
