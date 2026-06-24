#!/usr/bin/env python
"""Scrape Valorant comments via the Arctic Shift Reddit archive -> data/raw/comments.jsonl

Reddit's own API (public JSON *and* OAuth) blocks datacenter / flagged IPs, which broke
direct scraping. Arctic Shift (https://arctic-shift.photon-reddit.com) is a public Reddit
data archive (a Pushshift successor) on different infrastructure that is NOT blocked, with
a clean search API. No account, no app, no creds.

We sample across many time windows (ANCHORS) in both subreddits so the dataset isn't all
from one week / one drama, which keeps the label distribution realistic.

  RUN:
      uv run python scripts/scrape_arctic.py

Output: data/raw/comments.jsonl  (one comment per line; no labels yet).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
USER_AGENT = "script:takemeter:v0.1 (AI201 student project)"

SUBREDDITS = ["VALORANT", "ValorantCompetitive"]
# Anchor dates spread across ~2.5 years; we pull a few pages forward from each so the
# corpus spans many threads/seasons rather than clustering in one moment.
ANCHORS = [
    "2023-02-01", "2023-06-01", "2023-10-01",
    "2024-01-15", "2024-04-01", "2024-07-01", "2024-10-01",
    "2025-01-01", "2025-03-15", "2025-06-01",
]
PAGES_PER_ANCHOR = 1        # one page is enough once we cap per window
PER_PAGE = 100              # arctic-shift max
PER_WINDOW_CAP = 32         # cap each (sub, anchor) so the corpus spreads across all of
                            # them instead of clustering in the first few windows
# Confrontational terms used to *surface candidates* for the scarce `toxic` class.
# This only biases what we LOOK AT, not the label — many hits are neutral venting or
# critique and get labeled as such on review (those become the useful hard cases).
TOXIC_TERMS = [
    "trash", "garbage", "hardstuck", "uninstall", "clueless", "braindead",
    "washed", "clown", "cope", "ratio", "dogshit", "no skill", "bot frag",
]
PER_TERM_CAP = 8
MIN_CHARS = 15
MAX_CHARS = 1200
TARGET_TOTAL = 600          # collect generously; label/balance down to 200+
REQ_DELAY = 0.8

OUT = Path("data/raw/comments.jsonl")
SKIP_AUTHORS = {"AutoModerator", "[deleted]", "ValorantModTeam"}


def clean_light(text: str) -> str:
    return text.replace("​", "").strip()


def fetch(session, subreddit, after, before, body=None):
    if body:
        # full-text search: arctic-shift 422s if body is combined with date range +
        # sort, so query it on its own (subreddit + body + limit only).
        params = {"subreddit": subreddit, "body": body, "limit": PER_PAGE}
    else:
        params = {
            "subreddit": subreddit,
            "after": after,
            "before": before,
            "limit": PER_PAGE,
            "sort": "asc",
        }
    for attempt in range(4):
        r = session.get(API, params=params, timeout=40)
        if r.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json().get("data", [])
    return []


def keep(d, seen) -> bool:
    cid = d.get("id")
    body = d.get("body", "") or ""
    return bool(
        cid and cid not in seen
        and d.get("author", "") not in SKIP_AUTHORS
        and body not in ("[deleted]", "[removed]", "")
        and MIN_CHARS <= len(body) <= MAX_CHARS
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    comments: list[dict] = []
    seen: set[str] = set()

    # anchors outer, subs inner -> interleaves time periods AND both subreddits
    for anchor in ANCHORS:
        for sub in SUBREDDITS:
            if len(comments) >= TARGET_TOTAL:
                break
            after = anchor          # arctic-shift accepts YYYY-MM-DD or epoch
            kept_here = 0
            for _ in range(PAGES_PER_ANCHOR):
                if kept_here >= PER_WINDOW_CAP:
                    break
                try:
                    batch = fetch(session, sub, after, "2025-12-31")
                except Exception as e:
                    print(f"  warn: {sub}@{anchor} failed ({e})")
                    break
                if not batch:
                    break
                for d in batch:
                    if kept_here >= PER_WINDOW_CAP:
                        break
                    if keep(d, seen):
                        seen.add(d["id"])
                        kept_here += 1
                        parent = d.get("parent_id", "") or ""
                        comments.append({
                            "id": d["id"],
                            "text": clean_light(d["body"]),
                            "permalink": "https://www.reddit.com" + (d.get("permalink") or ""),
                            "subreddit": d.get("subreddit", sub),
                            "thread_id": (d.get("link_id", "") or "").replace("t3_", ""),
                            "thread_title": "",   # archive search omits title; not needed
                            "parent_id": parent,
                            "is_top_level": parent.startswith("t3_"),
                            "score": d.get("score", 0),
                            "created_utc": int(d.get("created_utc", 0)),
                            "scraped_utc": int(time.time()),
                        })
                after = int(batch[-1]["created_utc"]) + 1   # advance cursor
                time.sleep(REQ_DELAY)
            print(f"  {sub} @ {anchor}: +{kept_here} (total {len(comments)})")

    # ── toxic-class enrichment: surface heated candidates by text search ──────
    # Labels are decided on review, not by the search term — see TOXIC_TERMS note.
    print("\n  toxic-candidate enrichment:")
    for term in TOXIC_TERMS:
        kept_here = 0
        for sub in SUBREDDITS:
            try:
                batch = fetch(session, sub, "2023-01-01", "2025-12-31", body=term)
            except Exception as e:
                print(f"  warn: term '{term}'/{sub} failed ({e})")
                continue
            for d in batch:
                if kept_here >= PER_TERM_CAP:
                    break
                if keep(d, seen):
                    seen.add(d["id"])
                    kept_here += 1
                    parent = d.get("parent_id", "") or ""
                    comments.append({
                        "id": d["id"],
                        "text": clean_light(d["body"]),
                        "permalink": "https://www.reddit.com" + (d.get("permalink") or ""),
                        "subreddit": d.get("subreddit", sub),
                        "thread_id": (d.get("link_id", "") or "").replace("t3_", ""),
                        "thread_title": "",
                        "parent_id": parent,
                        "is_top_level": parent.startswith("t3_"),
                        "score": d.get("score", 0),
                        "created_utc": int(d.get("created_utc", 0)),
                        "scraped_utc": int(time.time()),
                        "search_term": term,   # provenance: how this candidate surfaced
                    })
            time.sleep(REQ_DELAY)
        print(f"    '{term}': +{kept_here}")

    with OUT.open("w", encoding="utf-8") as f:
        for c in comments:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_top = sum(c["is_top_level"] for c in comments)
    print(f"\n  wrote {len(comments)} comments -> {OUT}")
    print(f"  ({n_top} top-level, {len(comments) - n_top} replies)")
    if len(comments) < 250:
        print("  NOTE: under 250 — add more ANCHORS or raise PAGES_PER_ANCHOR, re-run.")


if __name__ == "__main__":
    main()
