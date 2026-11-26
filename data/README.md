# Data

We use the original LIAR v1.0 release (Wang, 2017), not LIAR-PLUS. It isn't committed here. Download it from Kaggle (`doanquanvietnamca/liar-dataset`) or from the original UCSB link, and put `train.tsv`, `valid.tsv`, `test.tsv` in `data/raw/`. (The old location, `data/` itself, still works.)

Check that your copy matches ours (the Kaggle copy and the original release are byte-identical):

| File | MD5 |
|---|---|
| train.tsv | `735058b28189e774d7a7a12334cb8219` |
| valid.tsv | `41e0cf170bbb85bb15eb6d3267c9cd22` |
| test.tsv | `394cd917733276391ceac6ac7542ebe0` |

The raw files have no header and 14 tab-separated columns. `src/data.py` loads them with our column names:

`id, label, statement, subject, speaker, job, state, party, barely_true_count, false_count, half_true_count, mostly_true_count, pants_on_fire_count, context`

It strips whitespace, fills missing metadata with `unknown`, lowercases label/party/job/state/context, and adds `<count>_corr` columns (the counts with the row's own label removed).

`data/splits/disjoint_{train,valid,test}.tsv` is the speaker-disjoint split (no speaker in more than one part). These files have a header, all the columns above plus the `_corr` columns, and are committed so everyone uses the same split. `src/make_splits.py` rebuilds them; `data.load_splits("disjoint")` gives the same rows. To read them without our code: `pd.read_csv(path, sep="\t", keep_default_na=False)`.

Sizes: train 10,270, valid 1,283, test 1,283 (official: 10,269 / 1,284 / 1,283). Label shares stay within 2.8 points of the official split in every part.
