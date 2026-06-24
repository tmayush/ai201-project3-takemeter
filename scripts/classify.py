#!/usr/bin/env python
"""Classify Valorant comments with the fine-tuned TakeMeter model — for the demo video.

Loads the trained DistilBERT checkpoint and prints, for each post, the predicted label
and the model's confidence (plus the full per-class probabilities). Use this on screen
in the demo video to show posts being classified live.

  RUN (built-in demo posts):
      uv run python scripts/classify.py

  RUN (your own post):
      uv run python scripts/classify.py "stop shooting while moving, counter-strafe first"
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ID_TO_LABEL = {0: "constructive", 1: "neutral", 2: "toxic"}
MODEL_DIR = "takemeter-model"

# Demo posts for the video. "expect" = my human label, for narration. The first three
# are clear-cut; the last two are the kinds of comment the model tends to miss.
DEMO_POSTS = [
    ("Stop shooting while moving — your bullets only land when you're fully stopped. "
     "Practice counter-strafing in the range for 10 min before you queue.", "constructive"),
    ("honestly ranked has felt so miserable this whole act lol", "neutral"),
    ("you're hardstuck iron for a reason, uninstall and stop queuing with us trash", "toxic"),
    ("he has 340k followers mostly from fortnite so he loses a lot every day", "toxic"),
    ("yea ur right my adr is dogshit lmfao hopefully ill improve that soon", "neutral"),
]


def latest_checkpoint(model_dir: str) -> str:
    ckpts = sorted(Path(model_dir).glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        return str(ckpts[-1])
    if (Path(model_dir) / "config.json").exists():
        return model_dir
    sys.exit(f"  ERROR: no trained model in {model_dir}/ — run scripts/run_local.py first.")


def main() -> None:
    path = latest_checkpoint(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    print(f"Loaded fine-tuned model from {path}\n" + "=" * 64)

    args = sys.argv[1:]
    posts = [(a, None) for a in args] if args else DEMO_POSTS

    for text, expect in posts:
        enc = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)[0]
        pred_id = int(probs.argmax())
        pred = ID_TO_LABEL[pred_id]
        conf = float(probs[pred_id])
        dist = "  ".join(f"{ID_TO_LABEL[i]} {float(probs[i]):.2f}" for i in range(3))

        print(f'\nPOST: "{text}"')
        print(f"  -> PREDICTED: {pred.upper()}   (confidence {conf:.0%})")
        print(f"     all classes: {dist}")
        if expect:
            mark = "correct" if pred == expect else f"WRONG (human label: {expect})"
            print(f"     [{mark}]")
    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
