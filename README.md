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
- `data/support_tickets.csv` — 70 labelled training examples.
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

The four `.joblib` model artifacts are generated locally by `train_model.py` and are intentionally not committed to GitHub. Run training once after installing the dependencies and whenever the training CSV or training code changes.

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
```

The evaluator prints ML-only intent/urgency accuracy and a separate rule-assisted urgency accuracy, then saves per-message predictions to `evaluation_results.csv`. The results file is generated locally and should be reviewed before sharing.

## Current limitations

- The training set is small and manually labelled; test-set size is only 30 examples, so metrics are noisy.
- Confidence scores come from the calibrated model, not from the rule layer. A rule-selected urgency is not itself a calibrated probability.
- The 30 held-out examples currently cover delivery, refund, payment, and account support; they do not test every intent class.
- Rule-assisted metrics can look better or worse depending on labelled examples. Do not treat rules as proof of generalization; inspect per-class reports and incorrect predictions.
- Suggested replies are drafts. A support agent must review them before sending.


## Urgency evaluation notes

The evaluator reports the raw ML urgency estimate separately from the rule-assisted final urgency, using the same shared rules as the app. This distinction matters because the probability/confidence displayed in the UI still comes from the ML model and is not a calibrated confidence score for a rule override. Review each misclassified row and keep test labels independent of model changes; do not change test labels just to raise the score.
