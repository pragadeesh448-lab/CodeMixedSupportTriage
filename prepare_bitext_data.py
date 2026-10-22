"""Download the official Bitext support dataset and map supported intents.

Run:
    python -m pip install datasets
    python prepare_bitext_data.py

The Bitext source contains English text. This script does not translate it or claim
that it contains Tanglish/Hinglish. Code-mixed examples are in codemixed_expanded.csv.
Urgency is not a native Bitext label; this script assigns weak heuristic labels.
"""
from pathlib import Path
import re
import pandas as pd
from datasets import load_dataset

OUT = Path("data/bitext_mapped.csv")
OUT.parent.mkdir(exist_ok=True)

# Dataset named in the NexAI PS 1.1 brief.
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
parts = []
for split_name in ds:
    frame = ds[split_name].to_pandas()
    parts.append(frame)
raw = pd.concat(parts, ignore_index=True).drop_duplicates()

# Bitext intent -> the app's coarser support-intent taxonomy.
INTENT_MAP = {
    "track_order": "delivery_issue",
    "check_order": "delivery_issue",
    "check_delivery_period": "delivery_issue",
    "delivery_options": "delivery_issue",
    "delivery_period": "delivery_issue",
    "track_refund": "refund_request",
    "get_refund": "refund_request",
    "check_refund_policy": "refund_request",
    "payment_issue": "payment_issue",
    "check_payment_methods": "payment_issue",
    "check_invoice": "account_support",
    "get_invoice": "account_support",
    "edit_account": "account_support",
    "registration_problems": "account_support",
    "recover_password": "account_support",
    "switch_account": "account_support",
    "create_account": "account_support",
    "delete_account": "account_support",
    "contact_customer_service": "account_support",
    "contact_human_agent": "account_support",
    "cancel_order": "order_cancellation",
    "change_order": "order_cancellation",
    "return_request": "return_request",
    "complaint": "product_complaint",
}

def col_name(names):
    lower = {str(c).lower(): c for c in names}
    for candidate in ("instruction", "text", "message", "query", "utterance"):
        if candidate in lower:
            return lower[candidate]
    return None

intent_col = next((c for c in raw.columns if str(c).lower() == "intent"), None)
text_col = col_name(raw.columns)
if intent_col is None or text_col is None:
    raise SystemExit(f"Could not identify text/intent columns. Dataset columns: {list(raw.columns)}")

raw["intent_key"] = raw[intent_col].astype(str).str.strip().str.lower()
raw["message"] = raw[text_col].astype(str).str.strip()
raw["intent"] = raw["intent_key"].map(INTENT_MAP)
raw = raw.dropna(subset=["intent"])
raw = raw[raw["message"].str.len() > 2]

def weak_urgency(message, intent):
    text = message.lower()
    high_terms = [
        "charged twice", "money was deducted", "payment failed", "fraud",
        "unauthorized", "account hacked", "urgent", "not received refund",
        "damaged", "broken", "locked out"
    ]
    low_terms = ["how do i", "how can i", "what is", "where can i", "please explain"]
    if any(term in text for term in high_terms):
        return "high"
    if any(term in text for term in low_terms):
        return "low"
    return "medium"

raw["urgency"] = [
    weak_urgency(message, intent)
    for message, intent in zip(raw["message"], raw["intent"])
]
out = raw[["message", "intent", "urgency"]].drop_duplicates(subset=["message", "intent"])
out.to_csv(OUT, index=False)
print(f"Saved {len(out):,} mapped English rows to {OUT}")
print("Intent distribution:\n", out["intent"].value_counts().to_string())
print("\nImportant: urgency labels are heuristic weak labels, not ground truth.")
