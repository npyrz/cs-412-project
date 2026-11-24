"""Check whether the *corrected* credit history still leaks the label for seen speakers.

LIAR's credit counts are one snapshot per speaker (identical on all of a speaker's rows).
correct_credit() subtracts the row's own label, so corrected = speaker_total - onehot(label).
From training rows (labels known) we can rebuild each speaker's raw total. For a test row
of a seen speaker, total - corrected is then a one-hot of the label (or all zero for `true`).
This script runs that lookup with no learning and reports test accuracy / macro-F1.
Output: results/leak_check.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data import COUNT_COLS, LABEL2COUNT, LABELS, correct_credit, load_split, speaker_disjoint

ROOT = Path(__file__).resolve().parent.parent
COUNT2LABEL = {v: k for k, v in LABEL2COUNT.items()}
FALLBACK = LABELS.index("half-true")  # majority class when the lookup can't decide


def lookup_attack(tr, te):
    """tr/te already corrected. Returns (pred, resolved_mask)."""
    totals = {}
    for r in tr.itertuples():
        if not r.speaker:
            continue
        v = np.array([getattr(r, c) for c in COUNT_COLS])
        if r.label in LABEL2COUNT:
            v[COUNT_COLS.index(LABEL2COUNT[r.label])] += 1
        totals.setdefault(r.speaker, set()).add(tuple(v))
    pred, resolved = [], []
    for r in te.itertuples():
        p = None
        t = totals.get(r.speaker)
        if t and len(t) == 1:
            d = np.array(next(iter(t))) - np.array([getattr(r, c) for c in COUNT_COLS])
            if (d == 0).all():
                p = LABELS.index("true")
            elif d.sum() == 1 and d.min() == 0:
                p = LABELS.index(COUNT2LABEL[COUNT_COLS[int(d.argmax())]])
        resolved.append(p is not None)
        pred.append(FALLBACK if p is None else p)
    return np.array(pred), np.array(resolved)


def main():
    tr, va, te = (load_split(s) for s in ["train", "valid", "test"])
    allrows = pd.concat([tr, va, te], ignore_index=True)
    named = allrows[allrows.speaker != ""]
    multi = named.groupby("speaker").filter(lambda g: len(g) > 1)
    const = (multi.groupby("speaker")[COUNT_COLS].nunique().max(axis=1) == 1).mean()
    out = {"history_constant_within_speaker": round(float(const), 4)}
    splits = {"official": (tr, te),
              "disjoint": (lambda s: (s[0], s[2]))(speaker_disjoint(allrows))}
    for name, (a, b) in splits.items():
        a, b = correct_credit(a), correct_credit(b)
        pred, res = lookup_attack(a, b)
        y = b["y"].to_numpy()
        out[name] = {"n_test": int(len(y)), "resolved_frac": round(float(res.mean()), 4),
                     "acc": float(accuracy_score(y, pred)),
                     "macro_f1": float(f1_score(y, pred, average="macro")),
                     "acc_on_resolved": float(accuracy_score(y[res], pred[res])) if res.any() else None}
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items()})
    (ROOT / "results" / "leak_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
