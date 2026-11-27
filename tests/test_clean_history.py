"""python -m pytest tests  (synthetic data, no LIAR files needed)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import CLEAN_COLS, LABEL2COUNT, LABELS, add_clean_history  # noqa: E402


def fake(n, speakers, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"speaker": rng.choice(speakers, n), "label": rng.choice(LABELS, n)})


def test_test_history_is_train_counts_and_ignores_test_labels():
    tr, va, te = fake(300, ["a", "b", "c"], 0), fake(40, ["a", "b", "z"], 1), fake(40, ["a", "c", "z"], 2)
    _, va1, te1 = add_clean_history(tr, va, te)
    for s in ["a", "c"]:
        want = [(tr[tr.speaker == s].label == lab).sum() for lab in LABEL2COUNT]
        assert (te1[te1.speaker == s][CLEAN_COLS].to_numpy() == want).all()
    assert (te1[te1.speaker == "z"][CLEAN_COLS].to_numpy() == 0).all()  # unseen speaker
    shuffled = [d.assign(label=d.label.sample(frac=1, random_state=3).to_numpy()) for d in (va, te)]
    _, va2, te2 = add_clean_history(tr, *shuffled)
    assert (va2[CLEAN_COLS].to_numpy() == va1[CLEAN_COLS].to_numpy()).all()
    assert (te2[CLEAN_COLS].to_numpy() == te1[CLEAN_COLS].to_numpy()).all()


def test_train_row_never_counts_its_own_label():
    # one speaker, every row 'false': each row's history must miss at least its own row
    tr = pd.DataFrame({"speaker": ["a"] * 50, "label": ["false"] * 50})
    tr1, _, _ = add_clean_history(tr, tr.head(1), tr.head(1), k=5)
    assert (tr1["false_count_clean"] <= 49).all() and (tr1["false_count_clean"] == 40).all()
    # a speaker with a single train row gets an empty history
    tr = pd.DataFrame({"speaker": ["solo", "b", "b"], "label": ["false", "true", "false"]})
    tr1, _, _ = add_clean_history(tr, tr.head(1), tr.head(1))
    assert (tr1[tr1.speaker == "solo"][CLEAN_COLS].to_numpy() == 0).all()
