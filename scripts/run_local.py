#!/usr/bin/env python
"""Local mirror of takemeter.ipynb — train + evaluate + Groq baseline, no Colab.

Runs the exact same pipeline as the notebook (same 70/15/15 stratified split with
random_state=42, same hyperparameters) on the local CUDA GPU, so the numbers match
what Colab would produce. Outputs the two artifacts the README needs:
    confusion_matrix.png
    evaluation_results.json

The notebook stays the canonical Colab deliverable; this is the fast path to real
metrics tonight (and a fallback if Colab misbehaves).

ENV: GROQ_API_KEY in .env (for the zero-shot baseline).
RUN: uv run python scripts/run_local.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import requests
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, Trainer, TrainingArguments, set_seed,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LABEL_MAP = {"constructive": 0, "neutral": 1, "toxic": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
MODEL_NAME = "distilbert-base-uncased"
CSV_PATH = "data/valorant_takes.csv"

# Same Valorant discourse-health prompt as the notebook's Groq cell.
SYSTEM_PROMPT = """You are classifying comments from the Valorant gaming community (Reddit).
Judge each comment by DISCOURSE HEALTH — how it contributes to the conversation —
NOT whether its opinion is correct.

Assign exactly one of these three labels:

constructive: Adds value — reasoned argument, helpful advice, thoughtful analysis, or a
respectful answer, even when disagreeing.
neutral: On-topic but low-substance and non-hostile — casual chatter, jokes, simple
reactions, one-line facts, plain questions, or undirected venting with no target.
toxic: Hostile or corrosive — insults, flaming, personal attacks, teammate-blaming as
abuse, slurs, or contemptuous put-downs.

Respond with ONLY the label name: constructive, neutral, or toxic.
Do not explain. Do not add punctuation."""


def main() -> None:
    set_seed(42)  # reproducible weight init / shuffling so reported numbers are stable
    print(f"GPU available: {torch.cuda.is_available()}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    df = pd.read_csv(CSV_PATH)
    df["label_id"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)
    print(f"Loaded {len(df)} examples; distribution:\n{df['label'].value_counts()}\n")

    # identical split to the notebook
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label_id"])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"])
    train_df, val_df, test_df = (d.reset_index(drop=True) for d in (train_df, val_df, test_df))
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=256)

    def make_ds(d):
        ds = Dataset.from_pandas(d[["text", "label_id"]].rename(columns={"label_id": "labels"}))
        return ds.map(tokenize, batched=True)

    train_ds, val_ds, test_ds = make_ds(train_df), make_ds(val_df), make_ds(test_df)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID_TO_LABEL, label2id=LABEL_MAP)

    def compute_metrics(p):
        return {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=-1))}

    # epochs=6 (not 3) and warmup_ratio (not warmup_steps=50): on 252 train examples a
    # batch of 16 is only ~16 steps/epoch, so 3 epochs = 48 steps < 50 warmup steps —
    # the LR never reached 2e-5 and the model badly underfit (toxic F1 was 0.00). A
    # 10% warmup ratio + 6 epochs lets it actually converge.
    args = TrainingArguments(
        output_dir="./takemeter-model",
        num_train_epochs=6, per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        logging_steps=10, report_to="none",
    )
    # class-weighted loss — toxic is the minority and was collapsing to F1=0; weighting
    # the loss by inverse class frequency forces the model to actually pay for missing it.
    counts = train_df["label_id"].value_counts().sort_index()
    weights = (len(train_df) / (NUM_LABELS * counts)).values
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"class weights (inv-freq): {dict(zip([ID_TO_LABEL[i] for i in range(NUM_LABELS)], weights.round(2)))}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=class_weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                              data_collator=collator, compute_metrics=compute_metrics)
    print("\nFine-tuning...")
    trainer.train()

    # ── evaluate fine-tuned ──
    out = trainer.predict(test_ds)
    ft_pred = np.argmax(out.predictions, axis=-1)
    ft_true = out.label_ids
    ft_acc = accuracy_score(ft_true, ft_pred)
    names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
    print(f"\nFine-tuned accuracy: {ft_acc:.3f}")
    print(classification_report(ft_true, ft_pred, target_names=names, zero_division=0))

    cm = confusion_matrix(ft_true, ft_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Fine-Tuned Model — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

    # wrong predictions for error analysis
    wrong = np.where(ft_pred != ft_true)[0]
    probs = torch.nn.functional.softmax(torch.tensor(out.predictions), dim=-1).numpy()
    print(f"\nWrong predictions: {len(wrong)}/{len(ft_true)} — first 12:")
    err_dump = []
    for idx in wrong[:12]:
        rec = {
            "text": test_df.iloc[idx]["text"],
            "true": ID_TO_LABEL[ft_true[idx]],
            "pred": ID_TO_LABEL[ft_pred[idx]],
            "confidence": round(float(probs[idx][ft_pred[idx]]), 3),
        }
        err_dump.append(rec)
        print(f"  [{rec['true']}->{rec['pred']} {rec['confidence']}] {rec['text'][:110]}")

    # ── Groq baseline ──
    bl_acc, n_valid = groq_baseline(test_df, ft_true)

    results = {
        "baseline_accuracy": round(bl_acc, 4),
        "finetuned_accuracy": round(ft_acc, 4),
        "improvement": round(ft_acc - bl_acc, 4),
        "test_set_size": int(len(test_df)),
        "baseline_parseable": int(n_valid),
        "label_map": LABEL_MAP,
        "model": MODEL_NAME,
        "per_class_finetuned": classification_report(
            ft_true, ft_pred, target_names=names, zero_division=0, output_dict=True),
        "sample_errors": err_dump,
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved evaluation_results.json")
    print(f"\n=== baseline {bl_acc:.3f} | fine-tuned {ft_acc:.3f} | "
          f"delta {ft_acc - bl_acc:+.3f} ===")


def groq_baseline(test_df, ft_true):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("\nNo GROQ_API_KEY — skipping baseline (baseline_accuracy=0).")
        return 0.0, 0
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    print(f"\nGroq baseline on {len(test_df)} test examples...")
    preds = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        label = None
        try:
            r = requests.post(url, headers=headers, timeout=40, json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify this post:\n\n{row['text']}"},
                ],
                "temperature": 0, "max_tokens": 20})
            if r.status_code == 200:
                raw = r.json()["choices"][0]["message"]["content"].strip().lower()
                for lab in sorted(LABEL_MAP, key=len, reverse=True):
                    if raw == lab or lab in raw:
                        label = lab
                        break
            elif r.status_code == 429:
                time.sleep(3)
        except Exception as e:
            print(f"  warn: {e}")
        preds.append(label)
        time.sleep(0.3)
    valid = [(LABEL_MAP[p], t) for p, t in zip(preds, ft_true) if p is not None]
    if not valid:
        return 0.0, 0
    bl_acc = accuracy_score([t for _, t in valid], [p for p, _ in valid])
    print(f"Baseline accuracy: {bl_acc:.3f} on {len(valid)}/{len(test_df)} parseable")
    return bl_acc, len(valid)


if __name__ == "__main__":
    main()
