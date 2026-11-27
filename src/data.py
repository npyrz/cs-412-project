"""LIAR loading, cleaning, credit-history leakage correction, and splits.

  python data.py          # label counts per file (quick check that loading works)

Files: data/raw/{train,valid,test}.tsv (LIAR v1, no header, 14 columns).
Cleaning: strip whitespace, missing metadata -> 'unknown', lowercase label/party/job/state/context.
Each row also gets <count>_corr columns: the credit counts with the row's own label removed.

load_splits() also adds <count>_clean columns: history rebuilt from training labels only
(see add_clean_history). Use these for any metadata run. LIAR's own counts are the
speaker's labels summed over the whole dataset, test set included, so both the raw and
the corrected counts leak test labels for speakers seen in training.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "splits"

COLS = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
        "barely_true_count", "false_count", "half_true_count", "mostly_true_count",
        "pants_on_fire_count", "context"]
LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
COUNT_COLS = ["pants_on_fire_count", "false_count", "barely_true_count", "half_true_count",
              "mostly_true_count"]
CORR_COLS = [c + "_corr" for c in COUNT_COLS]
CLEAN_COLS = [c + "_clean" for c in COUNT_COLS]
CLEAN_FOLDS = 5
# label -> credit-history column that counts it ("true" has no column in LIAR)
LABEL2COUNT = {"pants-fire": "pants_on_fire_count", "false": "false_count",
               "barely-true": "barely_true_count", "half-true": "half_true_count",
               "mostly-true": "mostly_true_count"}
CAT_COLS = ["speaker", "job", "state", "party", "context", "subject"]
LOWER_COLS = ["label", "party", "job", "state", "context"]
MISSING = "unknown"


def _raw_path(name):
    p = RAW_DIR / f"{name}.tsv"
    return p if p.exists() else ROOT / "data" / f"{name}.tsv"  # old layout


def load_split(name: str, lowercase: bool = True) -> pd.DataFrame:
    """One official split. lowercase=False keeps original case (only for reproducing old runs)."""
    df = pd.read_csv(_raw_path(name), sep="\t", header=None, names=COLS,
                     quoting=3, dtype=str, keep_default_na=False)
    for c in COUNT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in CAT_COLS + ["statement", "label"]:
        df[c] = df[c].fillna("").str.strip()
    for c in CAT_COLS:
        df[c] = df[c].replace("", MISSING)
    for c in LOWER_COLS if lowercase else ["label", "party"]:
        df[c] = df[c].str.lower()
    bad = set(df["label"]) - set(LABELS)
    assert not bad, f"{name}: unexpected labels {bad}"
    df["y"] = df["label"].map(LABEL2ID)
    df["y_bin"] = (df["y"] >= 3).astype(int)  # half-true and above = true-leaning
    df["split"] = name
    return add_corrected(df)


def load_all(lowercase: bool = True) -> pd.DataFrame:
    return pd.concat([load_split(s, lowercase) for s in ["train", "valid", "test"]],
                     ignore_index=True)


def add_corrected(df: pd.DataFrame) -> pd.DataFrame:
    """Add <count>_corr columns: the count matching the row's own label minus 1, never below 0."""
    for c in COUNT_COLS:
        df[c + "_corr"] = df[c]
    for lab, col in LABEL2COUNT.items():
        m = (df["label"] == lab) & (df[col] >= 1)
        df.loc[m, col + "_corr"] -= 1
    return df


def correct_credit(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with the corrected counts written over the raw count columns.

    For code that reads COUNT_COLS directly. New code should read CORR_COLS instead.
    """
    out = df.copy()
    out[COUNT_COLS] = out[CORR_COLS].to_numpy()
    return out


def add_clean_history(tr, va, te, k=CLEAN_FOLDS, seed=412):
    """Add <count>_clean columns built from training labels only. Returns copies.

    valid/test rows: counts of the speaker's labels in train (0 for unseen speakers).
    train rows: the same, but counting only train rows in the other k-1 folds, so a row's
    own label (and its fold-mates' labels) never appear in its own history.
    No valid or test label is used anywhere.
    """
    tr, va, te = tr.copy(), va.copy(), te.copy()
    onehot = pd.DataFrame(0, index=tr.index, columns=COUNT_COLS)
    for lab, col in LABEL2COUNT.items():
        onehot.loc[tr["label"] == lab, col] = 1
    spk = tr["speaker"].to_numpy()
    fold = np.random.default_rng(seed).permutation(len(tr)) % k
    total = onehot.groupby(spk).sum()
    in_fold = onehot.groupby([spk, fold]).sum()
    tr[CLEAN_COLS] = (total.loc[spk].to_numpy()
                      - in_fold.loc[list(zip(spk, fold))].to_numpy())
    for d in (va, te):
        d[CLEAN_COLS] = total.reindex(d["speaker"]).fillna(0).astype(int).to_numpy()
    return tr, va, te


def speaker_disjoint(df: pd.DataFrame, seed: int = 412):
    """Re-split all 12.8k statements so no speaker appears in more than one split.

    Target sizes roughly match the official 80/10/10 split.
    """
    sizes = df["speaker"].value_counts()
    rng = np.random.default_rng(seed)
    # The very largest speakers (e.g. barack-obama, ~600 rows) would swamp a 10% split,
    # so they always stay in train; everyone else is shuffled and assigned greedily.
    big = set(sizes[sizes > 0.02 * len(df)].index)
    order = [s for s in rng.permutation(sizes.index.to_numpy()) if s not in big]
    target = int(0.10 * len(df))
    assign, n_te, n_va = {}, 0, 0
    for s in order:
        if n_te < target:
            assign[s] = "test"; n_te += sizes[s]
        elif n_va < target:
            assign[s] = "valid"; n_va += sizes[s]
        else:
            assign[s] = "train"
    out = df.copy()
    out["split"] = df["speaker"].map(lambda s: assign.get(s, "train")).values
    return (out[out.split == "train"].copy(), out[out.split == "valid"].copy(),
            out[out.split == "test"].copy())


def load_splits(split: str = "official", lowercase: bool = True):
    """(train, valid, test) for 'official' or 'disjoint'.

    The disjoint split is rebuilt from the official files with speaker_disjoint(seed=412);
    it matches data/splits/disjoint_*.tsv row for row.
    """
    tr, va, te = (load_split(s, lowercase) for s in ["train", "valid", "test"])
    if split == "disjoint":
        tr, va, te = speaker_disjoint(pd.concat([tr, va, te], ignore_index=True))
    elif split != "official":
        raise ValueError(f"split must be 'official' or 'disjoint', got {split!r}")
    return add_clean_history(tr, va, te)


if __name__ == "__main__":
    df = load_all()
    print(pd.crosstab(df["label"], df["split"]).reindex(LABELS)[["train", "valid", "test"]])
    print("total", len(df))
