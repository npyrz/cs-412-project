"""Fine-tune BERT / RoBERTa on LIAR (milestones 3.3 and 3.4). Needs a GPU (Colab T4 is fine).

Examples:
  python finetune_transformer.py --model bert-base-uncased --meta none
  python finetune_transformer.py --model roberta-base --meta all --split disjoint
  python finetune_transformer.py --model roberta-large --meta all --drop credit --lr 1e-5
  python finetune_transformer.py --model roberta-base --meta all --task binary --seed 2

Metadata is serialized as text and prepended to the statement, with the
corrected credit history (own label removed) unless --raw-credit is passed.
Results go to results/transformers/<run-name>.json.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from datasets import Dataset

from data import LABELS, correct_credit, load_split, speaker_disjoint

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["party", "speaker", "job", "state", "context", "subject", "credit"]


def serialize(r, fields):
    bits = []
    if "speaker" in fields: bits.append(f"speaker: {r.speaker.replace('-', ' ')}")
    if "party" in fields: bits.append(f"party: {r.party}")
    if "job" in fields and r.job: bits.append(f"job: {r.job}")
    if "state" in fields and r.state: bits.append(f"state: {r.state}")
    if "context" in fields and r.context: bits.append(f"context: {r.context}")
    if "subject" in fields: bits.append(f"subject: {r.subject.replace(',', ', ')}")
    if "credit" in fields:
        bits.append(f"history: pants-fire {r.pants_fire_c}, false {r.false_c}, barely-true "
                    f"{r.barely_true_c}, half-true {r.half_true_c}, mostly-true {r.mostly_true_c}")
    return " | ".join(bits)


def build_text(df, fields):
    if not fields:
        return df["statement"].tolist()
    return [f"{serialize(r, fields)} || {r.statement}" for r in df.itertuples()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--meta", choices=["none", "all"], default="none")
    ap.add_argument("--drop", default="", help="comma list of metadata fields to ablate")
    ap.add_argument("--raw-credit", action="store_true")
    ap.add_argument("--split", choices=["official", "disjoint"], default="official")
    ap.add_argument("--task", choices=["6way", "binary"], default="6way")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1, help="effective batch = bs * grad-accum")
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=412)
    ap.add_argument("--out-dir", default="", help="where result JSON goes (default results/transformers)")
    ap.add_argument("--keep-checkpoints", action="store_true")
    a = ap.parse_args()

    tr, va, te = (load_split(s) for s in ["train", "valid", "test"])
    if a.split == "disjoint":
        tr, va, te = speaker_disjoint(pd.concat([tr, va, te], ignore_index=True))
    if not a.raw_credit:
        tr, va, te = (correct_credit(d) for d in (tr, va, te))
    fields = [] if a.meta == "none" else [f for f in FIELDS if f not in a.drop.split(",")]
    target = "y" if a.task == "6way" else "y_bin"
    labels = LABELS if a.task == "6way" else ["false-leaning", "true-leaning"]

    tok = AutoTokenizer.from_pretrained(a.model)
    def mk(df):
        ds = Dataset.from_dict({"text": build_text(df, fields), "label": df[target].tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_len), batched=True)
    dtr, dva, dte = mk(tr), mk(va), mk(te)

    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=len(labels))

    def compute(p):
        pred = p.predictions.argmax(-1)
        return {"acc": accuracy_score(p.label_ids, pred),
                "macro_f1": f1_score(p.label_ids, pred, average="macro")}

    name = (f"{a.model.split('/')[-1]}_{a.meta}{'-' + a.drop if a.drop else ''}"
            f"{'_rawcredit' if a.raw_credit else ''}_{a.split}_{a.task}_lr{a.lr:g}_s{a.seed}")
    args = TrainingArguments(
        output_dir=str(ROOT / "checkpoints" / name), learning_rate=a.lr,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
        per_device_eval_batch_size=64,
        num_train_epochs=a.epochs, weight_decay=0.01,
        warmup_steps=int(0.1 * a.epochs * -(-len(dtr) // (a.bs * a.grad_accum))),  # 10% warmup; works on transformers 4.x and 5.x
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", save_total_limit=1, seed=a.seed,
        fp16=torch.cuda.is_available(), report_to="none", logging_steps=50)
    trainer = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute)
    train_out = trainer.train()
    val_hist = [{"epoch": h["epoch"], "macro_f1": h["eval_macro_f1"], "acc": h["eval_acc"]}
                for h in trainer.state.log_history if "eval_macro_f1" in h]

    out = trainer.predict(dte)
    pred = out.predictions.argmax(-1)
    pr, rc, f1, _ = precision_recall_fscore_support(out.label_ids, pred,
                                                    labels=range(len(labels)), zero_division=0)
    res = {"run": name, "args": vars(a), "acc": accuracy_score(out.label_ids, pred),
           "macro_f1": f1_score(out.label_ids, pred, average="macro"),
           "weighted_f1": f1_score(out.label_ids, pred, average="weighted"),
           "per_class": {l: {"p": pr[i], "r": rc[i], "f1": f1[i]} for i, l in enumerate(labels)},
           "val_macro_f1": trainer.state.best_metric, "val_history": val_hist,
           "train_runtime_s": train_out.metrics.get("train_runtime"),
           "test_pred": pred.tolist()}
    d = Path(a.out_dir) if a.out_dir else ROOT / "results" / "transformers"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(res, indent=2, default=float))
    if not a.keep_checkpoints:
        shutil.rmtree(args.output_dir, ignore_errors=True)
    print(name, f"val_macroF1={res['val_macro_f1']:.4f} test acc={res['acc']:.4f} macroF1={res['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
