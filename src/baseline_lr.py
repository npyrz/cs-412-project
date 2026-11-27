"""TF-IDF + logistic regression baseline (final runs). About 5 minutes on CPU.

  python baseline_lr.py

Features
  text      TF-IDF of the statement, word 1-2 grams
  metadata  one-hot party, state, job, context (values seen < 5 times in train -> 'other'),
            plus the 5 history counts (log1p, standardized)

Runs, each on the official and the speaker-disjoint split, all logged with evaluate():
  text only                       history none
  text + metadata, raw counts     history raw        (leaks the row's own label)
  text + metadata, corrected      history corrected  (still leaks other test labels)
  text + metadata, clean          history clean      (training labels only; the honest one)
  ablations on the clean version: drop party / job / state / context / history, one at a time
C is picked on validation macro-F1. Writes results/runs.csv rows, results/predictions/,
and results/ablation_lr.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CLEAN_COLS, CORR_COLS, COUNT_COLS, load_splits
from evaluate import evaluate

RESULTS = Path(__file__).resolve().parent.parent / "results"
C_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
CATS = ["party", "state", "job", "context"]
FIELDS = CATS + ["history"]
HIST_COLS = {"raw": COUNT_COLS, "corrected": CORR_COLS, "clean": CLEAN_COLS}


def norm_context(s):
    # "a floor speech." / "a tv interview" -> "floor speech" / "tv interview"
    return (s.str.replace(r"[^a-z ]", " ", regex=True)
            .str.replace(r"\b(a|an|the)\b", " ", regex=True).str.split().str.join(" "))


class Features:
    def __init__(self, fields, history):
        self.cats = [f for f in CATS if f in fields]
        self.hist = HIST_COLS[history] if "history" in fields and history != "none" else None

    def _cat(self, df):
        f = df[self.cats].copy()
        if "context" in f:
            f["context"] = norm_context(f["context"])
        return f

    def fit(self, df):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                     max_features=50000, strip_accents="unicode").fit(df["statement"])
        if self.cats:
            self.ohe = OneHotEncoder(min_frequency=5, handle_unknown="infrequent_if_exist").fit(self._cat(df))
        if self.hist:
            self.scaler = StandardScaler().fit(np.log1p(df[self.hist].to_numpy(float)))
        return self

    def transform(self, df):
        parts = [self.tfidf.transform(df["statement"])]
        if self.cats:
            parts.append(self.ohe.transform(self._cat(df)))
        if self.hist:
            parts.append(sparse.csr_matrix(self.scaler.transform(np.log1p(df[self.hist].to_numpy(float)))))
        return sparse.hstack(parts).tocsr()


def fit_predict(tr, va, te, fields, history):
    feats = Features(fields, history).fit(tr)
    Xtr, Xva, Xte = (feats.transform(d) for d in (tr, va, te))
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, solver="saga", tol=1e-3, max_iter=2000,
                                 class_weight="balanced", random_state=0).fit(Xtr, tr["y"])
        f = f1_score(va["y"], clf.predict(Xva), average="macro")
        if best is None or f > best[0]:
            best = (f, C, clf)
    return best[2].predict(Xte), best[1], best[0]


def runs():
    """(input, history, dropped, fields) for every run on one split."""
    out = [("text", "none", "none", []),
           ("text+meta", "raw", "none", FIELDS),
           ("text+meta", "corrected", "none", FIELDS),
           ("text+meta", "clean", "none", FIELDS)]
    for f in FIELDS:
        out.append(("text+meta", "none" if f == "history" else "clean", f, [x for x in FIELDS if x != f]))
    return out


def main():
    rows = []
    for split in ["official", "disjoint"]:
        tr, va, te = load_splits(split)
        for inp, hist, dropped, fields in runs():
            pred, C, val = fit_predict(tr, va, te, fields, hist)
            info = {"model": "lr", "input": inp, "history": hist, "dropped": dropped,
                    "split": split, "shots": 0, "seed": 0}
            r = evaluate(info, te["id"], te["y"], pred)
            rows.append({**info, "C": C, "val_macro_f1": round(val, 4), "macro_f1": r["macro_f1"],
                         "accuracy": r["accuracy"], "binary_macro_f1": r["binary_macro_f1"]})

    df = pd.DataFrame(rows)
    print("\n" + df[["split", "input", "history", "dropped", "C", "val_macro_f1", "macro_f1",
                     "accuracy", "binary_macro_f1"]].to_string(index=False))

    # ablation table: full clean model vs. each field removed
    abl = []
    for split in ["official", "disjoint"]:
        d = df[(df.split == split) & (df.input == "text+meta")]
        full = d[(d.history == "clean") & (d.dropped == "none")].macro_f1.item()
        abl.append({"split": split, "dropped": "none", "macro_f1": full, "change": 0.0})
        for f in FIELDS:
            m = d[d.dropped == f].macro_f1.item()
            abl.append({"split": split, "dropped": f, "macro_f1": m, "change": round(m - full, 4)})
    abl = pd.DataFrame(abl)
    abl.to_csv(RESULTS / "ablation_lr.csv", index=False)
    print("\nAblation (clean history), change in macro-F1 when a field is removed:\n" + abl.to_string(index=False))


if __name__ == "__main__":
    main()
