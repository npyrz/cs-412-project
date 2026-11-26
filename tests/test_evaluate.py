"""python -m pytest tests"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import LABELS  # noqa: E402
from evaluate import COLUMNS, evaluate, parse_run_name, rebuild, run_name  # noqa: E402

INFO = {"model": "roberta-large", "input": "text+meta", "history": "corrected",
        "dropped": "none", "split": "disjoint", "shots": 0, "seed": 2}


def test_run_name_matches_shared_format():
    assert run_name(INFO) == "roberta-large_text+meta_corrected_none_disjoint_0_2"
    assert parse_run_name(run_name(INFO)) == INFO


def test_row_and_predictions_file(tmp_path):
    # 6 rows, 4 right. Errors: true->false (crosses the binary line), pants-fire->false (doesn't)
    y_true = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    y_pred = ["false", "false", "barely-true", "half-true", "mostly-true", "false"]
    row = evaluate(INFO, ids=["a", "b", "c", "d", "e", "f"], y_true=y_true, y_pred=y_pred, results=tmp_path)

    runs = pd.read_csv(tmp_path / "runs.csv")
    assert list(runs.columns) == COLUMNS and len(runs) == 1
    assert row["accuracy"] == pytest.approx(4 / 6)
    assert row["precision_false"] == pytest.approx(1 / 3) and row["recall_false"] == 1.0
    assert row["recall_pants-fire"] == 0.0 and row["recall_true"] == 0.0
    # macro-F1: pants-fire 0, false 0.5, barely/half/mostly 1, true 0
    assert row["macro_f1"] == pytest.approx(3.5 / 6)
    # binary: true-leaning recall 2/3, false-leaning all right -> F1s 0.8 and 6/7
    assert row["binary_macro_f1"] == pytest.approx((0.8 + 6 / 7) / 2)

    pred = pd.read_csv(tmp_path / "predictions" / f"{run_name(INFO)}.csv")
    assert list(pred.columns) == ["id", "true_label", "pred_label"]
    assert pred.true_label.tolist() == y_true and pred.pred_label.tolist() == y_pred


def test_int_labels_same_as_strings(tmp_path):
    t, p = [0, 1, 2, 3, 4, 5], [1, 1, 2, 3, 4, 1]
    a = evaluate(INFO, range(6), t, p, results=tmp_path)
    b = evaluate(INFO, range(6), [LABELS[i] for i in t], [LABELS[i] for i in p], results=tmp_path)
    assert a == b


def test_rerun_replaces_row_and_rebuild_matches(tmp_path):
    other = {**INFO, "input": "text", "history": "none", "seed": 3}
    evaluate(INFO, range(6), [0] * 6, [1] * 6, results=tmp_path)
    evaluate(other, range(6), [0] * 6, [0] * 6, results=tmp_path)
    evaluate(INFO, range(6), [0] * 6, [0] * 6, results=tmp_path)  # rerun of the first
    runs = pd.read_csv(tmp_path / "runs.csv")
    assert len(runs) == 2 and (runs.accuracy == 1.0).all()
    rebuilt = rebuild(tmp_path).sort_values("run_name").reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, runs.sort_values("run_name").reset_index(drop=True))


def test_bad_values_rejected(tmp_path):
    with pytest.raises(AssertionError):
        evaluate({**INFO, "split": "random"}, [1], ["true"], ["true"], results=tmp_path)
    with pytest.raises(AssertionError):
        evaluate(INFO, [1], ["very-true"], ["true"], results=tmp_path)
