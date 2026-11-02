# Data

We use the original LIAR v1.0 release (Wang, 2017). It isn't committed here. Download it from Kaggle (`doanquanvietnamca/liar-dataset`) or from the original UCSB link, and put `train.tsv`, `valid.tsv`, `test.tsv` in this folder.

Check that your copy matches ours (the Kaggle copy and the original release are byte-identical):

| File | MD5 |
|---|---|
| train.tsv | `735058b28189e774d7a7a12334cb8219` |
| valid.tsv | `41e0cf170bbb85bb15eb6d3267c9cd22` |
| test.tsv | `394cd917733276391ceac6ac7542ebe0` |

The files have no header and 14 tab-separated columns. `src/data.py` documents them.
