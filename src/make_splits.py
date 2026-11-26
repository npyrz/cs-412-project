"""Save the speaker-disjoint split to data/splits/disjoint_{train,valid,test}.tsv.

  python make_splits.py

The split is data.speaker_disjoint(seed=412): speakers with more than 2% of all rows
(Obama, Trump, Clinton) are kept in train, the rest are shuffled and assigned greedily
until valid and test each hold about 10% of rows. Files have a header, our column names
and the _corr columns. Prints sizes and label mix next to the official split.
"""
import pandas as pd

from data import COLS, CORR_COLS, LABELS, SPLIT_DIR, load_splits

PARTS = ["train", "valid", "test"]


def main():
    official = dict(zip(PARTS, load_splits("official")))
    disjoint = dict(zip(PARTS, load_splits("disjoint")))

    spk = {p: set(d.speaker) for p, d in disjoint.items()}
    overlap = (spk["train"] & spk["valid"]) | (spk["train"] & spk["test"]) | (spk["valid"] & spk["test"])
    assert not overlap, f"speakers in more than one part: {sorted(overlap)[:10]}"
    n = sum(len(d) for d in disjoint.values())
    assert n == sum(len(d) for d in official.values())

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for p, d in disjoint.items():
        d[COLS + CORR_COLS].to_csv(SPLIT_DIR / f"disjoint_{p}.tsv", sep="\t", index=False)

    print("Speaker overlap between parts: 0")
    print(f"{'':12}" + "".join(f"{s + ' ' + p:>18}" for p in PARTS for s in ["off", "dis"]))
    print(f"{'rows':12}" + "".join(f"{len(d[p]):>12} ({len(d[p]) / n:4.0%})"
                                   for p in PARTS for d in (official, disjoint)))
    print(f"{'speakers':12}" + "".join(f"{d[p].speaker.nunique():>18}" for p in PARTS for d in (official, disjoint)))
    gap = 0.0
    for lab in LABELS:
        pct = [(d[p].label == lab).mean() for p in PARTS for d in (official, disjoint)]
        gap = max(gap, *(abs(pct[i] - pct[i + 1]) for i in range(0, 6, 2)))
        print(f"{lab:12}" + "".join(f"{x:>18.1%}" for x in pct))
    print(f"Largest label-share gap, official vs disjoint, same part: {gap:.1%}")
    print(f"Wrote {SPLIT_DIR}/disjoint_{{train,valid,test}}.tsv")


if __name__ == "__main__":
    main()
