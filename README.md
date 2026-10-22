# Code-Mixed Support Triage Assistant

NEXAI Hack 2026 — Problem Statement 1.1

## Overview

A prototype customer-support triage app for messages written in Tanglish, Hinglish, or English. It predicts the support intent and urgency, shows model confidence, flags messages for human review, and suggests a reply draft.

## Project contents

- `app.py` — Streamlit dashboard, intent model, urgency rules, review flag and suggested reply.
- `train_model.py` — trains the TF-IDF / calibrated LinearSVC intent and urgency models.
- `triage_rules.py` — shared rule-based urgency and ambiguous-delivery review logic used by the dashboard, smoke tests, and evaluation.
- `test_model.py` — runs representative messages through the triage pipeline.
- `evaluate_model.py` — reports ML-only and rule-assisted metrics separately on the held-out test CSV.
- `data/support_tickets.csv` — original manually labelled training examples.
- `data/codemixed_expanded.csv` — 240 synthetic, labelled English/Tanglish/Hinglish examples.
- `data/bitext_mapped.csv` — generated locally from the official Bitext dataset by `prepare_bitext_data.py` (not committed).
- `data/test_cases.csv` — 30 labelled evaluation examples.

## Run on Windows

Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The four `.joblib` model artifacts are generated locally by `train_model.py` and are intentionally not committed to GitHub. The training script combines the original examples and the code-mixed examples; it also includes `data/bitext_mapped.csv` if present. Character n-grams are included to help with spelling variation in romanized language, but they do not guarantee perfect Tanglish/Hinglish classification.

### Prepare the official Bitext dataset

The NexAI problem statement recommends `bitext/Bitext-customer-support-llm-chatbot-training-dataset` from Hugging Face (about 27,000 English request/response pairs). The PDF describes the source dataset; it does not contain the full dataset itself. Run:

```powershell
python -m pip install datasets
python prepare_bitext_data.py
python train_model.py
```

`prepare_bitext_data.py` maps supported Bitext intents to this app's smaller intent taxonomy. The source is English, not Tanglish/Hinglish. Its urgency labels are weak heuristic labels because the Bitext dataset does not provide the app's urgency labels. Keep this limitation in reports and evaluate on a separately labelled held-out code-mixed test set.

If PowerShell blocks environment activation, use the current terminal's Python directly instead:
```powershell
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

## Evaluate

After training, run:
```powershell
python evaluate_model.py
python -m unittest discover -s tests
```

The evaluator prints ML-only intent/urgency accuracy and a separate rule-assisted urgency accuracy, then saves per-message predictions to `evaluation_results.csv`. The results file is generated locally and should be reviewed before sharing.

## Current limitations

- The training set is small and manually labelled; test-set size is only 30 examples, so metrics are noisy.
- Confidence scores come from the calibrated model, not from the rule layer. A rule-selected urgency is not itself a calibrated probability.
- The 30 held-out examples currently cover delivery, refund, payment, and account support; they do not test every intent class.
- Rule-assisted metrics can look better or worse depending on labelled examples. Do not treat rules as proof of generalization; inspect per-class reports and incorrect predictions.
- Suggested replies are drafts. A support agent must review them before sending.


## Urgency evaluation notes

The evaluator reports the raw ML urgency estimate separately from the rule-assisted final urgency, using the same shared rules as the app. The small unit-test suite checks high-risk payment/refund/account signals, routine low-urgency requests, and ambiguous delivery review. This distinction matters because the probability/confidence displayed in the UI still comes from the ML model and is not a calibrated confidence score for a rule override. Review each misclassified row and keep test labels independent of model changes; do not change test labels just to raise the score.


## Baseline evaluation snapshot

A local run of the current code and 30-row hold-out CSV reported **86.67% intent accuracy**, **26.67% ML-only urgency accuracy**, and **70.00% rule-assisted urgency accuracy**. This is one small prototype evaluation, not a guarantee of performance on future tickets. The rule-assisted score may be optimistic because rules were developed during iteration against these example messages; use a fresh, separately labelled test set for a less biased estimate.
