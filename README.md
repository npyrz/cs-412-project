# Text, Context, and Truth (CS 412)

Political misinformation detection on LIAR (Wang, 2017). We compare TF-IDF + logistic regression, fine-tuned BERT/RoBERTa, and prompted LLMs, each with and without speaker metadata. The research question: does speaker metadata help more than a bigger text model, and does any gain hold for speakers the model has never seen?

Team: Om Patel, Yusif Shawish, Noah Pyrzanowski. Application track.

Full result tables: [`results/RESULTS.md`](results/RESULTS.md) (generated from the JSON files, not typed by hand).

## Status

| Milestone | Status |
|---|---|
| 3.1 Data prep and analysis | Done. Shared loader, metadata line, scoring code and speaker-disjoint split files are in (Nov 25) |
| 3.2 TF-IDF + LR baseline | Done. Final runs (`baseline_lr.py`, Nov 27): 8 runs + 10 ablations on both splits in `results/runs.csv`, ablation table in `results/ablation_lr.csv` |
| 3.3 Transformers, text only | Done: BERT/RoBERTa base and large, LR grid for base models. RoBERTa-large 6-way diverged and needs a rerun |
| 3.4 Metadata in transformers | Done for all four models (raw and corrected credit), RoBERTa-base field ablation and disjoint split (3 seeds for RoBERTa-base). Corrected-credit numbers on the official split are leaky (see findings) |
| 3.5 Intermediate report | Written for Nov 1 (Section 4 text in `report/`) |
| 3.6 LLM zero/few-shot | Script ready (`src/llm_prompting.py`), not run |
| 3.7 Bias and generalization | Speaker-disjoint split saved in `data/splits/`, run for LR and RoBERTa-base. LR per-party errors done |
| 3.8 Seeds, CIs, figures | RoBERTa-base main grid has seeds 412-414. Other models are seed 412 only. No CIs or figures yet |

## Key findings so far

All test macro-F1, 6-way, seed 412 unless a mean over seeds is given. With 1,283 test rows, differences under about 0.03 are noise.

1. **Raw credit history leaks the label.** LIAR's credit counts include the statement being classified. Logistic regression goes from 0.254 (text) to 0.418 with raw metadata, but only to 0.266 once `data.correct_credit()` removes the row's own label. About 93% of the metadata gain is leakage.
2. **The correction is not enough for models that can memorize speakers.** The counts are one snapshot per speaker, so the corrected history is `speaker_total - onehot(label)`. Training labels reveal each speaker's total. A lookup with no learning (`src/leak_check.py`) uses this to label 83.6% of official test rows at 99.7% accuracy, for 0.872 macro-F1 overall. Leaving the speaker name out doesn't help: history plus party/job/state still labels 80.7% of rows (0.847 macro-F1), and history alone 57.0% (0.686). LR can't do this lookup. Transformers seem to partly learn it: RoBERTa-large with *corrected* credit scores 0.503, above its raw-credit run (0.481). On the speaker-disjoint split the lookup finds nothing.
   **Fix:** the counts turn out to be the speaker's labels summed over all of LIAR, test set included (96.9% of speakers match exactly, and no count comes from outside the dataset), so no subtraction can clean them. `data.add_clean_history()` rebuilds the history from training labels only (`history="clean"` in `metadata.py`, `--clean-credit` in `finetune_transformer.py`). On the clean history the same lookups fall to chance (macro-F1 0.06-0.10), and its columns don't change when test labels are shuffled. Metadata runs from now on should use it.
3. **With leak-free history, metadata adds nothing to LR.** Final LR runs (`results/runs.csv`): text 0.253, text + metadata with clean history 0.252 on the official split; 0.231 vs 0.224 on the disjoint split. Raw history gives 0.438 / 0.411, all leak. Every single-field ablation stays within 0.013, which is noise.
4. **Transformers beat TF-IDF on text alone, but model size doesn't help.** Text-only: LR 0.254, BERT-base 0.286, RoBERTa-base 0.284, BERT-large 0.283. Binary: 0.631 to 0.640 for every model.
5. **For unseen speakers, honest metadata adds nothing.** On the speaker-disjoint split, corrected metadata gives LR 0.225 vs 0.227 for text, and RoBERTa-base 0.265 ± 0.031 vs 0.281 ± 0.013 for text (mean ± SD over 3 seeds). On the official split the same RoBERTa-base comparison is 0.369 ± 0.013 vs 0.287 ± 0.004, and that gap is the leak. Raw credit still "helps" on the disjoint split (LR 0.406, RoBERTa-base 0.431) only because it contains the label.
6. **Leave-one-field-out results for RoBERTa-base are unstable.** Rerunning seed 412 with a different batch setup moved the full model from 0.315 to 0.381 and single ablations by up to 0.09. They aren't usable on the official split while history leaks. Only dropping credit brings it back to text level (0.295).

## What needs deciding next
- Rerun metadata runs with the clean history (`--clean-credit`). The corrected-history runs so far stay in the report only as the leaky reference. The speaker-disjoint split stays the test for unseen speakers (they have no history there).
- Rerun RoBERTa-large text-only 6-way (diverged: loss stayed above ln 6 and it predicts one class). Try another seed or lr 5e-6.
- Run the LLM experiments (3.6), then seeds 1-3 and bootstrap CIs for the final tables.

## Layout
- `data/raw/` LIAR TSVs go here (not committed, see `data/README.md` for source and MD5s)
- `data/splits/` speaker-disjoint split as TSVs (committed)
- `src/data.py` loading with our column names, `_corr` columns, `load_splits("official"|"disjoint")` (adds the leak-free `_clean` history)
- `src/metadata.py` the shared metadata line: `text_with_meta(row, fields, history)`
- `src/evaluate.py` shared scoring: `evaluate(run_info, ids, y_true, y_pred)` -> `results/runs.csv`, `results/predictions/<run_name>.csv`
- `src/make_splits.py` writes `data/splits/`
- `tests/` tests for the scoring code and the clean history (`python -m pytest tests`)
- `src/analyze_data.py` dataset stats -> `results/eda.txt`, `results/data_stats.json`, `results/label_dist.{pdf,png}`
- `src/baseline_lr.py` final LR runs (text, + meta with raw / corrected / clean history, ablations) -> `results/runs.csv`, `results/predictions/`, `results/ablation_lr.csv` (about 2 min on CPU)
- `src/baseline_lr_v1.py` the intermediate-report LR grid -> `results/lr_results.json`
- `src/finetune_transformer.py` one BERT/RoBERTa run -> `results/transformers/<run>.json` (val history and test predictions included)
- `src/run_grid.py` the full transformer grid, skips finished runs. Base models first, then large
- `src/leak_check.py` residual credit-history leak test -> `results/leak_check.json`
- `src/summarize_results.py` builds `results/RESULTS.md` from all result JSONs
- `src/llm_prompting.py` zero/few-shot LLM eval through any OpenAI-compatible API
- `notebooks/run_transformers_colab.ipynb` the grid on a free Colab T4
- `tools/train_gpu.bat` Windows setup (CUDA PyTorch in `.venv`) plus the grid on a local NVIDIA GPU. `tools/check_gpu.bat` prints GPU/driver info
- `results/transformer_grid_log.txt` timing and status of every grid run
- `docs/project_proposal.pdf` the proposal

Run names read `<model>_<meta>[-<dropped field>][_rawcredit]_<split>_<task>_lr<lr>_s<seed>`.

## Run
```
pip install -r requirements.txt
cd src
python data.py                      # label counts per file: checks the data loads
python analyze_data.py
python make_splits.py
python baseline_lr.py
python run_grid.py                  # GPU; or one run: python finetune_transformer.py --model roberta-base --meta all
python leak_check.py
python summarize_results.py
LLM_BASE_URL=... LLM_API_KEY=... python llm_prompting.py --model <name> --shots 0 --meta none
```

Transformer setup: 4 epochs, effective batch 32 (large: 16), max length 160, 10% warmup, weight decay 0.01, best epoch by validation macro-F1, seed 412. Base models use fp16, large models bf16. Base model LR picked from {1e-5, 2e-5, 3e-5} on validation; large models use 1e-5. The grid ran on an RTX 4060 laptop GPU (8 GB) on Nov 1-2, 2026, about 4 min per base run and 15 min per large run. RoBERTa-base seeds 413/414 and the seed-412 reruns of its metadata, ablation and disjoint runs (Nov 25) used micro-batch 16 with 2 accumulation steps, so the same effective batch of 32. The old seed-412 files are kept in `results/transformers/om_setup/`.

Metadata is serialized as text before the statement. The shared line (`src/metadata.py`, `--meta-format v2`), used from now on:
`party: democrat | job: state delegate | state: virginia | context: a floor speech. | history: barely-true 0, false 0, half-true 0, mostly-true 1, pants-fire 0 || statement: <statement>`

All transformer runs so far used the older line (`--meta-format v1`, still the default so reruns match):
`speaker: scott surovell | party: democrat | job: State delegate | state: Virginia | context: a floor speech. | subject: energy, history, job-accomplishments | history: pants-fire 0, false 0, barely-true 0, half-true 0, mostly-true 1 || <statement>`
