"""Shared scoring. Call evaluate() once after every run so all results are scored the same way.

    from evaluate import evaluate
    evaluate({"model": "roberta-large", "input": "text+meta", "history": "clean",
              "dropped": "none", "split": "disjoint", "shots": 0, "seed": 2},
             ids=test["id"], y_true=test["label"], y_pred=pred)

y_true / y_pred can be label strings or indices 0-5 in our label order.
It writes results/predictions/<run_name>.csv (id, true_label, pred_label) and puts one row
in results/runs.csv. Rerunning the same run_name replaces its row instead of adding a second.

If runs.csv gets a git merge conflict, keep either side and run
    python evaluate.py --rebuild
to rebuild it from results/predictions/.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from data import LABELS

RESULTS = Path(__file__).resolve().parent.parent / "results"
KEYS = ["model", "input", "history", "dropped", "split", "shots", "seed"]
ALLOWED = {"input": {"text", "text+meta"}, "history": {"clean", "raw", "corrected", "none"},
           "split": {"official", "disjoint"}}
COLUMNS = (["run_name"] + KEYS + ["accuracy", "macro_f1", "weighted_f1", "binary_macro_f1"]
           + [f"{m}_{l}" for l in LABELS for m in ("precision", "recall")])


def run_name(info):
    missing = [k for k in KEYS if k not in info]
    assert not missing, f"run_info is missing {missing}"
    for k, ok in ALLOWED.items():
        assert info[k] in ok, f"{k} must be one of {sorted(ok)}, got {info[k]!r}"
    assert "_" not in str(info["model"]), "model name can't contain '_' (run_name uses it as separator)"
    return "_".join(str(info[k]) for k in KEYS)


def parse_run_name(name):
    vals = name.rsplit("_", len(KEYS) - 1)
    info = dict(zip(KEYS, vals))
    info["shots"], info["seed"] = int(info["shots"]), int(info["seed"])
    return info


def to_ids(y):
    """Label strings or ints -> ints 0-5."""
    out = []
    for v in y:
        if isinstance(v, str) and not v.isdigit():
            assert v in LABELS, f"unknown label {v!r}"
            out.append(LABELS.index(v))
        else:
            assert 0 <= int(v) < len(LABELS), f"label index out of range: {v!r}"
            out.append(int(v))
    return out


def scores(y_true, y_pred):
    t, p = to_ids(y_true), to_ids(y_pred)
    row = {"accuracy": accuracy_score(t, p),
           "macro_f1": f1_score(t, p, average="macro", labels=range(6), zero_division=0),
           "weighted_f1": f1_score(t, p, average="weighted", labels=range(6), zero_division=0),
           # half-true and above = true-leaning
           "binary_macro_f1": f1_score([x >= 3 for x in t], [x >= 3 for x in p], average="macro",
                                       zero_division=0)}
    pr, rc, _, _ = precision_recall_fscore_support(t, p, labels=range(6), zero_division=0)
    for i, l in enumerate(LABELS):
        row[f"precision_{l}"], row[f"recall_{l}"] = pr[i], rc[i]
    return {k: round(float(v), 6) for k, v in row.items()}


def _write_runs(rows, results):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(results / "runs.csv", index=False)
    return df


def evaluate(run_info, ids, y_true, y_pred, results=RESULTS):
    """Score one run, save its predictions and its runs.csv row. Returns the row as a dict."""
    results = Path(results)
    name = run_name(run_info)
    ids, y_true, y_pred = list(ids), list(y_true), list(y_pred)
    assert len(ids) == len(y_true) == len(y_pred), "ids, y_true and y_pred must be the same length"
    assert len(set(ids)) == len(ids), "ids must be unique"

    (results / "predictions").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "true_label": [LABELS[i] for i in to_ids(y_true)],
                  "pred_label": [LABELS[i] for i in to_ids(y_pred)]}
                 ).to_csv(results / "predictions" / f"{name}.csv", index=False)

    row = {"run_name": name, **{k: run_info[k] for k in KEYS}, **scores(y_true, y_pred)}
    path = results / "runs.csv"
    old = pd.read_csv(path).to_dict("records") if path.exists() else []
    _write_runs([r for r in old if r["run_name"] != name] + [row], results)
    print(f"{name}: acc {row['accuracy']:.3f}, macro-F1 {row['macro_f1']:.3f}, "
          f"binary macro-F1 {row['binary_macro_f1']:.3f}")
    return row


def rebuild(results=RESULTS):
    """Recompute runs.csv from every file in results/predictions/."""
    results = Path(results)
    rows = []
    for f in sorted((results / "predictions").glob("*.csv")):
        p = pd.read_csv(f)
        rows.append({"run_name": f.stem, **parse_run_name(f.stem), **scores(p.true_label, p.pred_label)})
    df = _write_runs(rows, results)
    print(f"rebuilt {results / 'runs.csv'}: {len(df)} runs")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild runs.csv from results/predictions/")
    if ap.parse_args().rebuild:
        rebuild()
    else:
        ap.print_help()
