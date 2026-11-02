"""Run the transformer experiment grid locally (same phases as notebooks/run_transformers_colab.ipynb).

Skips any run whose result JSON already exists, so it is safe to stop and restart.
  python run_grid.py                 # all phases
  python run_grid.py --phases 1 2    # just some
Results: results/transformers/*.json. Log: logs/train.log
"""
import argparse, glob, json, subprocess, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "transformers"
LOG = ROOT / "logs" / "train.log"
SEED = 412
BASE = ["roberta-base", "bert-base-uncased"]
LARGE = ["roberta-large", "bert-large-uncased"]
FIELDS = ["party", "speaker", "job", "state", "context", "subject", "credit"]
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_name(model, meta="none", drop="", raw=False, split="official", task="6way", lr=2e-5, seed=SEED):
    return (f"{model.split('/')[-1]}_{meta}{'-' + drop if drop else ''}"
            f"{'_rawcredit' if raw else ''}_{split}_{task}_lr{lr:g}_s{seed}")


def run(model, lr, meta="none", drop="", raw=False, split="official", task="6way", seed=SEED):
    name = run_name(model, meta, drop, raw, split, task, lr, seed)
    if (OUT / f"{name}.json").exists():
        log(f"skip (done): {name}"); return
    # Effective batch: 32 for base, 16 for large. Split it up on small GPUs.
    if model in LARGE:
        bs, acc = (16, 1) if VRAM_GB >= 14 else (8, 2) if VRAM_GB >= 7 else (4, 4)
    else:
        bs, acc = (32, 1) if VRAM_GB >= 7 else (16, 2)
    cmd = [sys.executable, "finetune_transformer.py", "--model", model, "--lr", str(lr),
           "--bs", str(bs), "--grad-accum", str(acc), "--meta", meta, "--split", split,
           "--task", task, "--seed", str(seed), "--out-dir", str(OUT)]
    if drop: cmd += ["--drop", drop]
    if raw: cmd += ["--raw-credit"]
    log(f"RUN {name} (bs {bs} x accum {acc})")
    t = time.time()
    with open(ROOT / "logs" / f"{name}.log", "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=ROOT / "src", stdout=f, stderr=subprocess.STDOUT)
    if rc == 0 and (OUT / f"{name}.json").exists():
        r = json.loads((OUT / f"{name}.json").read_text())
        log(f"DONE {name} in {(time.time()-t)/60:.1f} min: val F1 {r['val_macro_f1']:.3f}, "
            f"test F1 {r['macro_f1']:.3f}, acc {r['acc']:.3f}")
    else:
        log(f"FAILED {name} (exit {rc}), see logs/{name}.log")


def best_lr(model):
    rs = [json.loads(Path(f).read_text()) for f in
          glob.glob(str(OUT / f"{model.split('/')[-1]}_none_official_6way_lr*_s{SEED}.json"))]
    if not rs: return 1e-5 if model in LARGE else 2e-5
    return max(rs, key=lambda r: r["val_macro_f1"])["args"]["lr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); LOG.parent.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU"
    log(f"start: {gpu}, {VRAM_GB:.1f} GB, phases {a.phases}")
    if 1 in a.phases:   # text only, 6-way, official, lr grid for base models
        for m in BASE:
            for lr in [1e-5, 2e-5, 3e-5]: run(m, lr)
        for m in LARGE: run(m, 1e-5)
    if 2 in a.phases:   # text + metadata, corrected and raw credit
        for m in BASE + LARGE:
            lr = best_lr(m); run(m, lr, meta="all"); run(m, lr, meta="all", raw=True)
    if 3 in a.phases:   # speaker-disjoint split
        m = "roberta-base"; lr = best_lr(m)
        run(m, lr, split="disjoint"); run(m, lr, meta="all", split="disjoint")
        run(m, lr, meta="all", raw=True, split="disjoint")
    if 4 in a.phases:   # binary, text only
        for m in BASE + LARGE: run(m, best_lr(m), task="binary")
    if 5 in a.phases:   # leave-one-field-out ablation
        m = "roberta-base"; lr = best_lr(m)
        for f in FIELDS: run(m, lr, meta="all", drop=f)
    log("ALL DONE")


if __name__ == "__main__":
    main()
