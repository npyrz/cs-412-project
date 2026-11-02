# Text, Context, and Truth (CS 412)

Political misinformation detection on LIAR (Wang, 2017). We compare TF-IDF + logistic regression, fine-tuned BERT/RoBERTa, and prompted LLMs, each with and without speaker metadata. Two things make the comparison fair: a fix for credit-history label leakage, and a speaker-disjoint split that tests whether models generalize to new speakers.

Team: Om Patel, Yusif Shawish, Noah Pyrzanowski. Application track.

## Key findings so far
- LIAR's credit-history counts include the statement's own label (the dataset README says so, and we verified it). `data.correct_credit()` removes it.
- Logistic regression, 6-way macro-F1: text 0.254, text + metadata (raw) 0.418, text + metadata (corrected) 0.266. About 93% of the metadata gain is leakage.
- On the speaker-disjoint split, corrected metadata adds nothing over text (0.225 vs 0.227).
- RoBERTa-base, text only, learning rate picked on validation: 0.284 test macro-F1, above the TF-IDF baseline at every learning rate tried (0.283 to 0.295).

## Layout
- `data/` LIAR TSVs go here (not committed, see `data/README.md`)
- `src/data.py` loading, cleaning, credit-history correction, speaker-disjoint split
- `src/analyze_data.py` dataset stats -> `results/data_stats.json` and the label distribution plot
- `src/baseline_lr.py` LR grid, field ablation, per-party errors -> `results/lr_results.json` (about 5 min on CPU)
- `src/finetune_transformer.py` one BERT/RoBERTa fine-tuning run -> `results/transformers/<run>.json` (includes val history and test predictions)
- `src/run_grid.py` runs the full transformer grid in priority order, skipping finished runs
- `src/llm_prompting.py` zero/few-shot LLM eval through any OpenAI-compatible API
- `notebooks/run_transformers_colab.ipynb` the same grid on a free Colab T4, results saved to Google Drive
- `tools/train_gpu.bat` one-click Windows setup (CUDA PyTorch in `.venv`) plus the grid on a local NVIDIA GPU
- `report/what_we_have_done_so_far.txt` intermediate report Section 4

## Run
```
pip install -r requirements.txt
cd src
python analyze_data.py
python baseline_lr.py
python finetune_transformer.py --model roberta-base --meta none        # one run, needs a GPU
python run_grid.py --phases 1 2                                         # or the grid
LLM_BASE_URL=... LLM_API_KEY=... python llm_prompting.py --model <name> --shots 0 --meta none
```

Transformer grid (`run_grid.py`):
1. Text only, 6-way, official split. BERT-base and RoBERTa-base at lr 1e-5 / 2e-5 / 3e-5; BERT-large and RoBERTa-large at 1e-5.
2. Text + metadata (corrected and raw credit), each model at its best lr from phase 1.
3. Speaker-disjoint split, RoBERTa-base.
4. Binary task, text only.
5. Leave-one-field-out metadata ablation, RoBERTa-base.

Defaults: 4 epochs, batch 32 (large: effective 16), max length 160, 10% warmup, weight decay 0.01, fp16, best epoch by validation macro-F1, seed 412.
