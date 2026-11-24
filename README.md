# Text, Context, and Truth (CS 412)

Political misinformation detection on LIAR (Wang, 2017). We compare TF-IDF + logistic regression, fine-tuned BERT/RoBERTa, and prompted LLMs, each with and without speaker metadata. The research question: does speaker metadata help more than a bigger text model, and does any gain hold for speakers the model has never seen?

Team: Om Patel, Yusif Shawish, Noah Pyrzanowski. Application track.

Full result tables: [`results/RESULTS.md`](results/RESULTS.md) (generated from the JSON files, not typed by hand).

## Status

| Milestone | Status |
|---|---|
| 3.1 Data prep and analysis | Done |
| 3.2 TF-IDF + LR baseline | Done (official and speaker-disjoint splits, 6-way and binary, field ablation, per-party errors) |
| 3.3 Transformers, text only | Done: BERT/RoBERTa base and large, LR grid for base models. RoBERTa-large 6-way diverged and needs a rerun |
| 3.4 Metadata in transformers | Done for all four models (raw and corrected credit), RoBERTa-base field ablation and disjoint split. Corrected-credit numbers on the official split are leaky (see findings) |
| 3.5 Intermediate report | Written for Nov 1 (Section 4 text in `report/`) |
| 3.6 LLM zero/few-shot | Script ready (`src/llm_prompting.py`), not run |
| 3.7 Bias and generalization | Speaker-disjoint split done for LR and RoBERTa-base. LR per-party errors done |
| 3.8 Seeds, CIs, figures | Not started (all runs are seed 412, single run) |

## Key findings so far

All test macro-F1, 6-way, seed 412. With 1,283 test rows, differences under about 0.03 are noise.

1. **Raw credit history leaks the label.** LIAR's credit counts include the statement being classified. Logistic regression goes from 0.254 (text) to 0.418 with raw metadata, but only to 0.266 once `data.correct_credit()` removes the row's own label. About 93% of the metadata gain is leakage.
2. **The correction is not enough for models that can memorize speakers.** The counts are one snapshot per speaker, so the corrected history is `speaker_total - onehot(label)`. Training labels reveal each speaker's total. A lookup with no learning (`src/leak_check.py`) uses this to label 83.6% of official test rows at 99.7% accuracy, for 0.872 macro-F1 overall. LR can't do this lookup. Transformers seem to partly learn it: RoBERTa-large with *corrected* credit scores 0.503, above its raw-credit run (0.481). On the speaker-disjoint split the lookup finds nothing.
3. **Transformers beat TF-IDF on text alone, but model size doesn't help.** Text-only: LR 0.254, BERT-base 0.286, RoBERTa-base 0.284, BERT-large 0.283. Binary: 0.631 to 0.640 for every model.
4. **For unseen speakers, honest metadata adds nothing.** On the speaker-disjoint split, corrected metadata gives LR 0.225 vs 0.227 for text, and RoBERTa-base 0.230 vs 0.281 for text (worse). Raw credit still "helps" there (0.406 and 0.424) only because it contains the label.
5. **Leave-one-field-out results for RoBERTa-base are unstable** (0.334 to 0.429 against a full-model 0.315) and are not usable until the residual leak is removed. Only dropping credit brings it back to text level (0.299).

## What needs deciding next
- How to handle credit history for the final report: drop it, report metadata gains only on the speaker-disjoint split, or rebuild histories from earlier statements only.
- Rerun RoBERTa-large text-only 6-way (diverged: loss stayed above ln 6 and it predicts one class). Try another seed or lr 5e-6.
- Run the LLM experiments (3.6), then seeds 1-3 and bootstrap CIs for the final tables.

## Layout
- `data/` LIAR TSVs go here (not committed, see `data/README.md` for source and MD5s)
- `src/data.py` loading, cleaning, `correct_credit()`, `speaker_disjoint()`
- `src/analyze_data.py` dataset stats -> `results/data_stats.json`, `results/label_dist.{pdf,png}`
- `src/baseline_lr.py` LR grid, field ablation, per-party errors -> `results/lr_results.json` (about 5 min on CPU)
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
python analyze_data.py
python baseline_lr.py
python run_grid.py                  # GPU; or one run: python finetune_transformer.py --model roberta-base --meta all
python leak_check.py
python summarize_results.py
LLM_BASE_URL=... LLM_API_KEY=... python llm_prompting.py --model <name> --shots 0 --meta none
```

Transformer setup: 4 epochs, effective batch 32 (large: 16), max length 160, 10% warmup, weight decay 0.01, best epoch by validation macro-F1, seed 412. Base models use fp16, large models bf16. Base model LR picked from {1e-5, 2e-5, 3e-5} on validation; large models use 1e-5. The grid ran on an RTX 4060 laptop GPU (8 GB) on Nov 1-2, 2026, about 4 min per base run and 15 min per large run.

Metadata is serialized as text before the statement:
`speaker: scott surovell | party: democrat | job: State delegate | state: Virginia | context: a floor speech. | subject: energy, history, job-accomplishments | history: pants-fire 0, false 0, barely-true 0, half-true 0, mostly-true 1 || <statement>`
