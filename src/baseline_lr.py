"""TF-IDF + logistic regression baselines (milestone 3.2).

Conditions (each on official and speaker-disjoint splits, 6-class and binary):
  text            TF-IDF of the statement only
  text+meta(raw)  + all metadata, credit history as released (includes own label)
  text+meta(corr) + all metadata, own label removed from credit history
  text+meta(-cr)  + all metadata except credit history
  meta(corr)      metadata only, corrected credit history
Also: leave-one-field-out ablation and per-party error rates.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import OneHotEncoder

from data import COUNT_COLS, LABELS, correct_credit, load_split, speaker_disjoint

OUT = Path(__file__).resolve().parent.parent / "results"
SEED = 412
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
META_FIELDS = ["party", "speaker", "job", "state", "context", "subject", "credit"]


def norm_context(s: pd.Series) -> pd.Series:
    # "a floor speech." / "a TV interview" -> lowercase, drop articles and punctuation
    return (s.str.lower().str.replace(r"[^a-z ]", " ", regex=True)
            .str.replace(r"\b(a|an|the)\b", " ", regex=True)
            .str.split().str.join(" "))


def credit_feats(df: pd.DataFrame) -> np.ndarray:
    c = df[COUNT_COLS].to_numpy(float)
    tot = c.sum(1, keepdims=True)
    prop = np.divide(c, tot, out=np.zeros_like(c), where=tot > 0)
    return np.hstack([prop, np.log1p(tot), (tot == 0).astype(float)])


class Featurizer:
    def __init__(self, fields):
        self.fields = set(fields)

    def fit(self, df):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                     max_features=50000, strip_accents="unicode")
        self.tfidf.fit(df["statement"])
        cats = [f for f in ["party", "speaker", "job", "state", "context"] if f in self.fields]
        self.cats = cats
        if cats:
            self.ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
            self.ohe.fit(self._cat_frame(df))
        if "subject" in self.fields:
            self.subj = TfidfVectorizer(tokenizer=lambda s: [t for t in s.split(",") if t],
                                        token_pattern=None, lowercase=True, use_idf=False, norm=None)
            self.subj.fit(df["subject"])
        return self

    def _cat_frame(self, df):
        f = df[self.cats].copy()
        if "context" in f:
            f["context"] = norm_context(f["context"])
        if "job" in f:
            f["job"] = f["job"].str.lower()
        return f

    def transform(self, df, use_text=True):
        parts = []
        if use_text:
            parts.append(self.tfidf.transform(df["statement"]))
        if self.cats:
            parts.append(self.ohe.transform(self._cat_frame(df)))
        if "subject" in self.fields:
            parts.append(self.subj.transform(df["subject"]))
        if "credit" in self.fields:
            parts.append(sparse.csr_matrix(credit_feats(df)))
        return sparse.hstack(parts).tocsr()


def metrics(y, p, labels):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=range(len(labels)), zero_division=0)
    return {"acc": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
            "weighted_f1": f1_score(y, p, average="weighted"),
            "per_class": {l: {"p": pr[i], "r": rc[i], "f1": f1[i]} for i, l in enumerate(labels)}}


def run(tr, va, te, fields, use_text=True, target="y"):
    feat = Featurizer(fields).fit(tr)
    Xtr, Xva, Xte = (feat.transform(d, use_text) for d in (tr, va, te))
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, solver="saga", tol=1e-3, max_iter=2000,
                                 class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, tr[target])
        f = f1_score(va[target], clf.predict(Xva), average="macro")
        if best is None or f > best[0]:
            best = (f, C, clf)
    _, C, clf = best
    pred = clf.predict(Xte)
    labels = LABELS if target == "y" else ["false-leaning", "true-leaning"]
    m = metrics(te[target], pred, labels)
    m["C"], m["val_macro_f1"] = C, best[0]
    return m, pred


CONDITIONS = {
    "text": dict(fields=[], use_text=True, corr=False),
    "text+meta(raw)": dict(fields=META_FIELDS, use_text=True, corr=False),
    "text+meta(corr)": dict(fields=META_FIELDS, use_text=True, corr=True),
    "text+meta(-credit)": dict(fields=[f for f in META_FIELDS if f != "credit"], use_text=True, corr=True),
    "meta(corr)": dict(fields=META_FIELDS, use_text=False, corr=True),
}


def main():
    train, valid, test = (load_split(s) for s in ["train", "valid", "test"])
    full = pd.concat([train, valid, test], ignore_index=True)
    splits = {"official": (train, valid, test), "disjoint": speaker_disjoint(full)}
    results, preds = {}, {}
    for sname, (tr, va, te) in splits.items():
        for target in ["y", "y_bin"]:
            for cname, cfg in CONDITIONS.items():
                d = [correct_credit(x) if cfg["corr"] else x for x in (tr, va, te)]
                m, p = run(*d, cfg["fields"], cfg["use_text"], target)
                key = f"{sname}|{'6way' if target == 'y' else 'binary'}|{cname}"
                results[key] = m
                preds[key] = p
                print(f"{key:45s} acc={m['acc']:.3f} macroF1={m['macro_f1']:.3f} "
                      f"wF1={m['weighted_f1']:.3f} C={m['C']}", flush=True)

    # Leave-one-field-out ablation: official split, 6-way, corrected credit
    tr, va, te = (correct_credit(x) for x in (train, valid, test))
    abl = {}
    for f in META_FIELDS:
        m, _ = run(tr, va, te, [x for x in META_FIELDS if x != f], True, "y")
        abl[f"-{f}"] = m
        print(f"ablation -{f:10s} macroF1={m['macro_f1']:.3f} acc={m['acc']:.3f}", flush=True)
    for f in META_FIELDS:
        m, _ = run(tr, va, te, [f], True, "y")
        abl[f"+{f}"] = m
        print(f"ablation text+{f:8s} macroF1={m['macro_f1']:.3f} acc={m['acc']:.3f}", flush=True)
    results["ablation_official_6way_corr"] = abl

    # Per-party error rates on official test (6-way)
    party = test["party"].where(test["party"].isin(["republican", "democrat", "none"]), "other")
    by_party = {}
    for cname in ["text", "text+meta(raw)", "text+meta(corr)"]:
        p = preds[f"official|6way|{cname}"]
        err = pd.Series(p != test["y"].to_numpy(), index=test.index)
        perr = pd.Series(preds[f"official|binary|{cname}"] != test["y_bin"].to_numpy(), index=test.index)
        by_party[cname] = {g: {"n": int((party == g).sum()),
                               "err_6way": float(err[party == g].mean()),
                               "err_binary": float(perr[party == g].mean())}
                           for g in ["republican", "democrat", "none", "other"]}
    results["party_errors"] = by_party

    # Confusion matrices for the main official 6-way conditions
    results["confusion"] = {c: confusion_matrix(test["y"], preds[f"official|6way|{c}"]).tolist()
                            for c in ["text", "text+meta(raw)", "text+meta(corr)"]}

    (OUT / "lr_results.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
