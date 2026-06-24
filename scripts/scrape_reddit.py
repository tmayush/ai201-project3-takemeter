#!/usr/bin/env python
"""Scrape Valorant comments -> data/raw/comments.jsonl

NO REDDIT APP NEEDED. By default this hits Reddit's public JSON endpoints
(`www.reddit.com/...json`) with a polite User-Agent and slow request rate. That works
fine from a normal home/residential IP — the bot-blocking that hit Claude's dev
environment is IP-specific to datacenters, not you.

  RUN:
      uv run python scripts/scrape_reddit.py

If — and only if — you get repeated HTTP 403s (your IP is blocked too), set up the
official OAuth API as a fallback: create a "script" app at
https://www.reddit.com/prefs/apps (the CLASSIC page — NOT developers.reddit.com, that's
the unrelated Devvit platform), then put these in a .env at the repo root:
      REDDIT_CLIENT_ID=xxxx
      REDDIT_CLIENT_SECRET=xxxx
The script auto-detects them and switches to oauth.reddit.com.

Output: data/raw/comments.jsonl  (one comment per line; no labels yet).
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

# ── config ────────────────────────────────────────────────────────────────
SUBREDDITS = ["VALORANT", "ValorantCompetitive"]
# (listing, time_filter) — "controversial" is deliberate: it surfaces the heated
# threads that yield the scarce `toxic` class. See planning.md distribution note.
LISTINGS = [
    ("hot", None),
    ("top", "month"),
    ("controversial", "month"),
    ("controversial", "week"),
]
THREADS_PER_LISTING = 25
COMMENTS_PER_THREAD = 18
MIN_CHARS = 15          # drop "lol" / "ggs" noise floor; keep short-but-real takes
MAX_CHARS = 1200        # drop essays/copypasta
TARGET_TOTAL = 500      # stop once we have enough raw to label down to 200+
SEED_THREADS = [        # the original AI-sourced beginner threads
    "1m7ln5m", "1q6cubg", "1srepds", "1tg93yb", "kmv2y0",
]
# A descriptive UA with contact info is what Reddit's rules ask for; it markedly
# reduces the chance of a block on public endpoints.
USER_AGENT = "script:takemeter:v0.1 (AI201 student project)"
REQ_DELAY = 1.5         # seconds between requests on the public API (be polite)

OUT = Path("data/raw/comments.jsonl")
SKIP_AUTHORS = {"AutoModerator", "[deleted]", "ValorantModTeam"}


def die(msg: str) -> None:
    print(f"\n  ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ── transport: public JSON by default, OAuth only if creds are present ────
class Reddit:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        cid = os.environ.get("REDDIT_CLIENT_ID")
        secret = os.environ.get("REDDIT_CLIENT_SECRET")
        if cid and secret:
            self.base = "https://oauth.reddit.com"
            self.delay = 0.7
            self._auth(cid, secret)
            print("  mode: OAuth (oauth.reddit.com)")
        else:
            self.base = "https://www.reddit.com"
            self.delay = REQ_DELAY
            print("  mode: public JSON (no app/creds) — slower but no setup")

    def _auth(self, cid: str, secret: str) -> None:
        r = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(cid, secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        if r.status_code != 200:
            die(f"OAuth token request failed: HTTP {r.status_code} — {r.text[:200]}")
        self.session.headers["Authorization"] = f"Bearer {r.json()['access_token']}"

    def get(self, path: str, **params):
        # public endpoints need the .json suffix; oauth ones must NOT have it
        if self.base.endswith("reddit.com") and "oauth" not in self.base:
            path = path + ".json"
        url = f"{self.base}{path}"
        for attempt in range(4):
            r = self.session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 403:
                die("HTTP 403 from Reddit — your IP appears blocked on the public API.\n"
                    "  Fallback: create a CLASSIC 'script' app at "
                    "https://www.reddit.com/prefs/apps (scroll to bottom, 'create an "
                    "app...'), put REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET in .env, re-run.\n"
                    "  Do NOT use developers.reddit.com — that's the wrong (Devvit) platform.")
            r.raise_for_status()
            time.sleep(self.delay)
            return r.json()
        die(f"repeatedly rate-limited on {path}")


def discover_threads(api: Reddit) -> list[tuple[str, str, str]]:
    """Return list of (thread_id, subreddit, title), deduped, seeds first."""
    found: dict[str, tuple[str, str, str]] = {}
    for tid in SEED_THREADS:
        found[tid] = (tid, "VALORANT", "(seed thread)")
    for sub in SUBREDDITS:
        for listing, tf in LISTINGS:
            params = {"limit": THREADS_PER_LISTING}
            if tf:
                params["t"] = tf
            try:
                data = api.get(f"/r/{sub}/{listing}", **params)
            except SystemExit:
                raise
            except Exception as e:
                print(f"  warn: {sub}/{listing} failed ({e}); skipping")
                continue
            for child in data["data"]["children"]:
                d = child["data"]
                tid = d["id"]
                if tid not in found:
                    found[tid] = (tid, sub, d.get("title", ""))
    print(f"  discovered {len(found)} unique threads")
    return list(found.values())


def clean_light(text: str) -> str:
    # light only — heavy cleaning happens in build_csv.py so raw stays faithful
    return text.replace("​", "").strip()


def walk_comments(node, thread_id, subreddit, title, out, seen, per_thread):
    """Recursively collect real comments from an already-loaded comment tree."""
    for child in node.get("data", {}).get("children", []):
        if per_thread[0] >= COMMENTS_PER_THREAD:
            return
        if child.get("kind") != "t1":  # t1 = comment; "more" stubs skipped
            continue
        d = child["data"]
        cid = d.get("id")
        body = d.get("body", "") or ""
        author = d.get("author", "")
        if (
            cid and cid not in seen
            and author not in SKIP_AUTHORS
            and body not in ("[deleted]", "[removed]", "")
            and not d.get("stickied")
            and MIN_CHARS <= len(body) <= MAX_CHARS
        ):
            seen.add(cid)
            per_thread[0] += 1
            parent = d.get("parent_id", "")
            out.append({
                "id": cid,
                "text": clean_light(body),
                "permalink": "https://www.reddit.com" + d.get("permalink", ""),
                "subreddit": subreddit,
                "thread_id": thread_id,
                "thread_title": title,
                "parent_id": parent,
                "is_top_level": parent.startswith("t3_"),
                "score": d.get("score", 0),
                "created_utc": int(d.get("created_utc", 0)),
                "scraped_utc": int(time.time()),
            })
        replies = d.get("replies")
        if isinstance(replies, dict):
            walk_comments(replies, thread_id, subreddit, title, out, seen, per_thread)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    api = Reddit()

    threads = discover_threads(api)
    comments: list[dict] = []
    seen: set[str] = set()

    for i, (tid, sub, title) in enumerate(threads, 1):
        if len(comments) >= TARGET_TOTAL:
            break
        try:
            data = api.get(f"/r/{sub}/comments/{tid}", limit=200, depth=4, sort="top")
        except SystemExit:
            raise
        except Exception as e:
            print(f"  warn: thread {tid} failed ({e}); skipping")
            continue
        listing = data[1] if isinstance(data, list) and len(data) > 1 else {}
        if not title or title == "(seed thread)":
            try:
                title = data[0]["data"]["children"][0]["data"]["title"]
            except Exception:
                pass
        per_thread = [0]
        walk_comments(listing, tid, sub, title, comments, seen, per_thread)
        print(f"  [{i}/{len(threads)}] {sub}/{tid}: +{per_thread[0]} (total {len(comments)})")

    with OUT.open("w", encoding="utf-8") as f:
        for c in comments:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_top = sum(c["is_top_level"] for c in comments)
    print(f"\n  wrote {len(comments)} comments -> {OUT}")
    print(f"  ({n_top} top-level, {len(comments) - n_top} replies)")
    if len(comments) < 250:
        print("  NOTE: under 250 raw comments. Bump THREADS_PER_LISTING / "
              "COMMENTS_PER_THREAD or add thread IDs to SEED_THREADS, then re-run.")


if __name__ == "__main__":
    main()
