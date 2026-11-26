"""The shared metadata line. Transformers and LLM prompts all build their input with this.

  party: republican | job: u.s. senator | state: texas | context: a debate | history: barely-true 3, false 5, half-true 2, mostly-true 1, pants-fire 0 || statement: <the statement>

Usage:
  from data import load_splits
  from metadata import text_with_meta, text_only
  tr, va, te = load_splits("disjoint")
  x = [text_with_meta(r) for r in tr.itertuples()]                          # all fields, corrected history
  x = [text_with_meta(r, fields=fields_without("party")) for r in tr.itertuples()]  # ablation
  x = [text_with_meta(r, history="raw") for r in tr.itertuples()]           # leaky raw counts

Rows can be pandas rows (iterrows / itertuples) or dicts with our column names.
  python metadata.py    # prints example lines
"""
FIELDS = ("party", "job", "state", "context", "history")
# history order as in the shared format (alphabetical by label)
HISTORY = [("barely-true", "barely_true_count"), ("false", "false_count"),
           ("half-true", "half_true_count"), ("mostly-true", "mostly_true_count"),
           ("pants-fire", "pants_on_fire_count")]


def fields_without(field):
    """All fields except one, for leave-one-out ablations."""
    assert field in FIELDS, f"unknown field {field!r}, expected one of {FIELDS}"
    return tuple(f for f in FIELDS if f != field)


def _get(row, key):
    return getattr(row, key) if isinstance(row, tuple) else row[key]  # itertuples vs Series/dict


def _val(row, key):
    v = str(_get(row, key)).strip().lower()
    return v if v else "unknown"


def serialize(row, fields=FIELDS, history="corrected"):
    """Metadata line for one row. Fields not in `fields` are left out entirely.

    history: 'corrected' (own label removed, the _corr columns), 'raw' (as released, leaks
    the label) or 'none' (same as leaving 'history' out of fields).
    """
    assert history in ("corrected", "raw", "none"), history
    unknown = set(fields) - set(FIELDS)
    assert not unknown, f"unknown fields {unknown}, expected some of {FIELDS}"
    bits = [f"{f}: {_val(row, f)}" for f in FIELDS if f in fields and f != "history"]
    if "history" in fields and history != "none":
        suffix = "_corr" if history == "corrected" else ""
        bits.append("history: " + ", ".join(f"{lab} {int(_get(row, col + suffix))}"
                                            for lab, col in HISTORY))
    return " | ".join(bits)


def text_with_meta(row, fields=FIELDS, history="corrected"):
    meta = serialize(row, fields, history)
    stmt = str(_get(row, "statement"))
    return f"{meta} || statement: {stmt}" if meta else text_only(row)


def text_only(row):
    return str(_get(row, "statement"))


if __name__ == "__main__":
    from data import load_split
    rows = load_split("train").head(3)
    for r in rows.itertuples():
        print("all fields:     ", text_with_meta(r))
        print("without party:  ", text_with_meta(r, fields_without("party")))
        print("without history:", text_with_meta(r, fields_without("history")))
        print("raw history:    ", serialize(r, history="raw"))
        print()
