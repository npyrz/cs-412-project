"""Run the transformer experiment grid locally (or on Kaggle/Colab).

Skips any run whose result JSON already exists, so it is safe to stop and restart.
  python run_grid.py --dry-run                     # list what is left, train nothing
  python run_grid.py --models roberta-base         # laptop
  python run_grid.py --models roberta-large        # Kaggle T4
  python run_grid.py --phases 2 --seeds 413        # just some phases / seeds

Phases:
  1. LR pick: text only, official split, seed 412 (base: 1e-5/2e-5/3e-5, large: 1e-5)
  2. Main grid: {text, text+meta} x {official, disjoint} x seeds (all settings for one seed
     finish before the next seed starts)
  3. Leak demo: text+meta with raw credit counts, both splits, seed 412
  4. Ablation: roberta-base, text+meta, official, drop one field at a time, seed 412
  5. Binary task, text only, official, seed 412 (optional, not in the default)
Results: results/transformers/*.json. Log: logs/train.log
"""
import argparse, glob, json, subprocess, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "transformers"
LOG = ROOT / "logs" / "train.log"
SEED = 412  # LR selection, leak demo, ablation and binary use this seed only
BASE = ["roberta-base", "bert-base-uncased"]
LARGE = ["roberta-large", "bert-large-uncased"]
DEFAULT_MODELS = ["roberta-base", "roberta-large"]
FIELDS = ["party", "speaker", "job", "state", "context", "subject", "credit"]
COLLAPSE_F1 = 0.15  # validation macro-F1 below this means the model predicts ~one class
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
        if r["val_macro_f1"] < COLLAPSE_F1:
            log(f"WARNING {name} collapsed (val F1 {r['val_macro_f1']:.3f} < {COLLAPSE_F1}): "
                f"delete results/transformers/{name}.json and rerun")
    else:
        log(f"FAILED {name} (exit {rc}), see logs/{name}.log")


def best_lr(model):
    """Learning rate with the best validation macro-F1 among phase-1 runs (seed 412)."""
    rs = [json.loads(Path(f).read_text()) for f in
          glob.glob(str(OUT / f"{model.split('/')[-1]}_none_official_6way_lr*_s{SEED}.json"))]
    if not rs: return 1e-5 if model in LARGE else 2e-5
    return max(rs, key=lambda r: r["val_macro_f1"])["args"]["lr"]


def plan(phases, models, seeds):
    """Ordered list of run kwargs. Base models come before large ones within each phase."""
    models = sorted(models, key=lambda m: m in LARGE)
    runs = []
    if 1 in phases:
        for m in models:
            for lr in ([1e-5, 2e-5, 3e-5] if m in BASE else [1e-5]):
                runs.append(dict(model=m, lr=lr))
    if 2 in phases:
        for s in seeds:
            for m in models:
                for meta in ["none", "all"]:
                    for split in ["official", "disjoint"]:
                        runs.append(dict(model=m, meta=meta, split=split, seed=s))
    if 3 in phases:
        for m in models:
            for split in ["official", "disjoint"]:
                runs.append(dict(model=m, meta="all", raw=True, split=split))
    if 4 in phases and "roberta-base" in models:
        for f in FIELDS:
            runs.append(dict(model="roberta-base", meta="all", drop=f))
    if 5 in phases:
        for m in models:
            runs.append(dict(model=m, task="binary"))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS, choices=BASE + LARGE)
    ap.add_argument("--seeds", nargs="*", type=int, default=[412, 413, 414],
                    help="seeds for the main grid (phase 2)")
    ap.add_argument("--dry-run", action="store_true", help="list runs that are left, train nothing")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); LOG.parent.mkdir(parents=True, exist_ok=True)

    todo, done = [], 0
    for kw in plan(a.phases, a.models, a.seeds):
        # Phase-1 runs carry their own lr; everything else uses the validation-picked lr.
        # (Computed lazily so phase-2 runs see the LR results from phase 1 of this session.)
        lr = kw.pop("lr", None)
        todo.append((lr, kw))

    if a.dry_run:
        left = []
        for lr, kw in todo:
            name = run_name(lr=lr if lr is not None else best_lr(kw["model"]), **kw)
            if (OUT / f"{name}.json").exists(): done += 1
            else: left.append(name)
        print(f"{done} done, {len(left)} left:")
        for n in left: print("  " + n)
        return

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU"
    log(f"start: {gpu}, {VRAM_GB:.1f} GB, phases {a.phases}, models {a.models}, seeds {a.seeds}")
    for lr, kw in todo:
        lr = lr if lr is not None else best_lr(kw["model"])
        if (OUT / f"{run_name(lr=lr, **kw)}.json").exists():
            continue  # already done; skip quietly
        run(lr=lr, **kw)
    log("ALL DONE")


if __name__ == "__main__":
    main()
