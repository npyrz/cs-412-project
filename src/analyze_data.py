"""Dataset analysis for the intermediate report (milestone 3.1)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import LABELS, LABEL2COUNT, COUNT_COLS, load_all, speaker_disjoint

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)
df = load_all()
res = {}

# 1. sizes and class balance
res["sizes"] = df["split"].value_counts().to_dict()
dist = pd.crosstab(df["label"], df["split"]).reindex(LABELS)[["train", "valid", "test"]]
res["label_dist"] = dist.to_dict()
res["label_pct_train"] = (dist["train"] / dist["train"].sum() * 100).round(1).to_dict()

# 2. missing metadata
miss = {c: round(float((df[c] == "").mean() * 100), 2)
        for c in ["speaker", "job", "state", "party", "context", "subject"]}
res["missing_pct"] = miss
res["party_dist"] = df["party"].replace("", "(missing)").value_counts().head(8).to_dict()
res["n_party_values"] = int(df["party"].nunique())

# 3. duplicates
norm = df["statement"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
df["norm"] = norm
dup_within = int(norm.duplicated(keep=False).sum())
tr_set = set(norm[df.split == "train"])
res["dup_rows_any"] = dup_within
res["test_in_train_exact"] = int(norm[df.split == "test"].isin(tr_set).sum())
res["valid_in_train_exact"] = int(norm[df.split == "valid"].isin(tr_set).sum())
g = df.groupby("norm")["label"].nunique()
res["dup_statements_conflicting_labels"] = int((g > 1).sum())

# 4. speakers
spk = df[df.speaker != ""]
counts = spk["speaker"].value_counts()
res["n_speakers"] = int(counts.size)
res["top_speakers"] = counts.head(5).to_dict()
res["speakers_with_1_stmt"] = int((counts == 1).sum())
res["share_rows_top10_speakers"] = round(float(counts.head(10).sum() / len(df) * 100), 1)
tr_spk = set(df.loc[df.split == "train", "speaker"])
te = df[df.split == "test"]
res["test_rows_seen_speaker_pct"] = round(float(te["speaker"].isin(tr_spk).mean() * 100), 1)

# 5. credit-history leakage check
# For every label that has a count column, is that count >= 1 for the row itself?
leak = {}
for lab, col in LABEL2COUNT.items():
    sub = df[df.label == lab]
    leak[lab] = round(float((sub[col] >= 1).mean() * 100), 1)
res["pct_rows_own_label_count_ge1"] = leak
# Baseline for comparison: how often is a *different* label's count >= 1 for the same rows?
other = {}
for lab, col in LABEL2COUNT.items():
    sub = df[df.label == lab]
    others = [c for c in COUNT_COLS if c != col]
    other[lab] = round(float((sub[others] >= 1).mean().mean() * 100), 1)
res["pct_rows_other_label_count_ge1"] = other
# Single-statement speakers: is their history exactly a one-hot of their own label?
one = df[df["speaker"].map(df["speaker"].value_counts()) == 1]
one = one[one.label != "true"]
hist = one[COUNT_COLS].sum(axis=1)
own = one.apply(lambda r: r[LABEL2COUNT[r.label]], axis=1)
res["single_stmt_speakers_checked"] = int(len(one))
res["single_stmt_history_equals_own_label_pct"] = round(float(((hist == 1) & (own == 1)).mean() * 100), 1)
res["single_stmt_history_all_zero_pct"] = round(float((hist == 0).mean() * 100), 1)
res["rows_history_all_zero_pct"] = round(float((df[COUNT_COLS].sum(axis=1) == 0).mean() * 100), 1)

# 6. speaker-disjoint split
tr, va, tst = speaker_disjoint(df)
res["disjoint_sizes"] = {"train": len(tr), "valid": len(va), "test": len(tst)}
res["disjoint_overlap_check"] = len(set(tr.speaker) & set(tst.speaker) - {""})
res["disjoint_test_label_pct"] = (tst["label"].value_counts(normalize=True).reindex(LABELS) * 100).round(1).to_dict()

# 7. statement length
words = df["statement"].str.split().str.len()
res["stmt_words_mean"] = round(float(words.mean()), 1)
res["stmt_words_median"] = int(words.median())

(OUT / "data_stats.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))

# Figure: class distribution
fig, ax = plt.subplots(figsize=(4.2, 2.4))
pct = dist.div(dist.sum(axis=0), axis=1) * 100
pct.plot.bar(ax=ax, width=0.8, color=["#3b6ea8", "#8fb3d9", "#d98f3b"])
ax.set_ylabel("% of split"); ax.set_xlabel("")
ax.tick_params(axis="x", labelrotation=30, labelsize=8)
ax.legend(fontsize=7, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "label_dist.pdf"); fig.savefig(OUT / "label_dist.png", dpi=150)
