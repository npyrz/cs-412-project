"""LIAR loading, cleaning, credit-history leakage correction, and splits."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COLS = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
        "barely_true_c", "false_c", "half_true_c", "mostly_true_c", "pants_fire_c", "context"]
LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
COUNT_COLS = ["pants_fire_c", "false_c", "barely_true_c", "half_true_c", "mostly_true_c"]
# label -> credit-history column that counts it ("true" has no column in LIAR)
LABEL2COUNT = {"pants-fire": "pants_fire_c", "false": "false_c", "barely-true": "barely_true_c",
               "half-true": "half_true_c", "mostly-true": "mostly_true_c"}
CAT_COLS = ["speaker", "job", "state", "party", "context", "subject"]


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.tsv", sep="\t", header=None, names=COLS,
                     quoting=3, dtype=str, keep_default_na=False)
    for c in COUNT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in CAT_COLS + ["statement"]:
        df[c] = df[c].fillna("").str.strip()
    df["party"] = df["party"].str.lower()
    df["y"] = df["label"].map(LABEL2ID)
    df["y_bin"] = (df["y"] >= 3).astype(int)  # half-true and above = true-leaning
    df["split"] = name
    return df


def load_all() -> pd.DataFrame:
    return pd.concat([load_split(s) for s in ["train", "valid", "test"]], ignore_index=True)


def correct_credit(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the current statement's own label removed from its credit history.

    Only subtracts when the matching count is >= 1, so rows whose history was
    collected before the statement was added are left alone.
    """
    out = df.copy()
    for lab, col in LABEL2COUNT.items():
        m = (out["label"] == lab) & (out[col] >= 1)
        out.loc[m, col] -= 1
    return out


def speaker_disjoint(df: pd.DataFrame, seed: int = 412):
    """Re-split all 12.8k statements so no speaker appears in more than one split.

    Target sizes roughly match the official 80/10/10 split.
    """
    groups = df["speaker"].replace("", "__unknown__")
    sizes = groups.value_counts()
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
    split = groups.map(lambda s: assign.get(s, "train"))
    out = df.copy()
    out["split"] = split.values
    return (out[out.split == "train"].copy(), out[out.split == "valid"].copy(),
            out[out.split == "test"].copy())
