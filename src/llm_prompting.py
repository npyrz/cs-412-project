"""Zero-/few-shot LLM evaluation on LIAR (milestone 3.6).

Uses any OpenAI-compatible chat endpoint (OpenAI, Together, Groq, a local Ollama
server, ...). Set LLM_BASE_URL, LLM_API_KEY and pass --model.

  python llm_prompting.py --model gpt-4o-mini --shots 0 --meta none
  python llm_prompting.py --model gpt-4o-mini --shots 6 --meta all --limit 300

Prompts are fixed and temperature is 0 so runs are reproducible. Few-shot
examples are drawn once from train with a fixed seed (one per class).
Responses are cached in results/llm/cache_<model>.jsonl so reruns are free.
"""
import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score

from data import LABELS, correct_credit, load_split
from finetune_transformer import serialize

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["party", "speaker", "job", "state", "context", "subject", "credit"]

SYSTEM = ("You are a careful political fact-checker. Rate the truthfulness of the statement "
          "using exactly one PolitiFact label: pants-fire, false, barely-true, half-true, "
          "mostly-true, true. Answer with the label only.")


def fmt(r, fields):
    meta = serialize(r, fields) if fields else ""
    return (f"Speaker info: {meta}\n" if meta else "") + f"Statement: {r.statement}\nLabel:"


def parse(text):
    t = text.strip().lower()
    for lab in sorted(LABELS, key=len, reverse=True):  # match "half-true" before "true"
        if re.search(rf"\b{re.escape(lab)}\b", t):
            return LABELS.index(lab)
    return LABELS.index("half-true")  # fallback: majority class; count these separately


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--meta", choices=["none", "all"], default="none")
    ap.add_argument("--limit", type=int, default=0, help="evaluate on a fixed random subset")
    a = ap.parse_args()

    client = OpenAI(base_url=os.environ.get("LLM_BASE_URL"), api_key=os.environ.get("LLM_API_KEY"))
    tr, te = correct_credit(load_split("train")), correct_credit(load_split("test"))
    if a.limit:
        te = te.sample(a.limit, random_state=412)
    fields = FIELDS if a.meta == "all" else []

    shots = []
    if a.shots:
        per = max(1, a.shots // len(LABELS))
        ex = tr.groupby("label", group_keys=False).apply(lambda g: g.sample(per, random_state=412))
        for r in ex.sample(frac=1, random_state=412).itertuples():
            shots += [{"role": "user", "content": fmt(r, fields)},
                      {"role": "assistant", "content": r.label}]

    out_dir = ROOT / "results" / "llm"; out_dir.mkdir(parents=True, exist_ok=True)
    cache_f = out_dir / f"cache_{a.model.replace('/', '_')}.jsonl"
    cache = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            d = json.loads(line); cache[d["key"]] = d["text"]

    preds, unparsed = [], 0
    with cache_f.open("a") as fh:
        for r in te.itertuples():
            msgs = [{"role": "system", "content": SYSTEM}] + shots + [{"role": "user", "content": fmt(r, fields)}]
            key = f"{a.shots}|{a.meta}|{r.id}"
            if key not in cache:
                resp = client.chat.completions.create(model=a.model, messages=msgs,
                                                      temperature=0, max_tokens=8)
                cache[key] = resp.choices[0].message.content or ""
                fh.write(json.dumps({"key": key, "text": cache[key]}) + "\n"); fh.flush()
            if not any(l in cache[key].lower() for l in LABELS):
                unparsed += 1
            preds.append(parse(cache[key]))

    y = te["y"].to_numpy()
    res = {"model": a.model, "shots": a.shots, "meta": a.meta, "n": len(te), "unparsed": unparsed,
           "acc": accuracy_score(y, preds), "macro_f1": f1_score(y, preds, average="macro"),
           "weighted_f1": f1_score(y, preds, average="weighted")}
    name = f"{a.model.replace('/', '_')}_{a.shots}shot_{a.meta}"
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=2, default=float))
    print(res)


if __name__ == "__main__":
    main()
