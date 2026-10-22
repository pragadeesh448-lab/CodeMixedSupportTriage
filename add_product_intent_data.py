import csv
from pathlib import Path

path = Path("data/codemixed_expanded.csv")

new_rows = [
    ("vera product vanduruku", "product_complaint", "medium"),
    ("vera item vandhuruku", "product_complaint", "medium"),
    ("wrong product vandhuruku", "product_complaint", "medium"),
    ("product maathi vandhuruku", "product_complaint", "medium"),
    ("macha item seri illada", "product_complaint", "medium"),
    ("I received a different product from what I ordered", "product_complaint", "medium"),
    ("The product delivered is not the one I ordered", "product_complaint", "medium"),
    ("I got the wrong model", "product_complaint", "medium"),
    ("product exchange pannanum", "product_exchange", "medium"),
    ("indha item exchange pannunga", "product_exchange", "medium"),
    ("wrong product vandhurukku exchange venum", "product_exchange", "medium"),
    ("size correct illa, exchange pannanum", "product_exchange", "medium"),
    ("I want to exchange this item", "product_exchange", "medium"),
    ("Can you send me a replacement?", "product_exchange", "medium"),
    ("mujhe product exchange karna hai", "product_exchange", "medium"),
    ("indha product details sollunga", "product_inquiry", "low"),
    ("indha phone price enna", "product_inquiry", "low"),
    ("product stock la irukka", "product_inquiry", "low"),
    ("How much does this product cost?", "product_inquiry", "low"),
    ("is product ki details batao", "product_inquiry", "low"),
    ("product return pannanum", "return_request", "medium"),
    ("wrong product vandhurukku return pannanum", "return_request", "medium"),
    ("I want to return this product", "return_request", "medium"),
    ("enakku refund venum", "refund_request", "medium"),
    ("please give me a refund for this item", "refund_request", "medium"),
    ("paise wapas chahiye", "refund_request", "medium"),
]

if not path.exists():
    raise SystemExit(f"Dataset not found: {path}. Run this in your project root.")

with path.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)

required = {"message", "intent", "urgency"}
if not fields or not required.issubset(fields):
    raise SystemExit(f"Unexpected CSV columns: {fields}")

existing = {(r["message"].strip().casefold(), r["intent"].strip()) for r in rows}
added = 0

for message, intent, urgency in new_rows:
    key = (message.casefold(), intent)
    if key not in existing:
        rows.append({"message": message, "intent": intent, "urgency": urgency})
        existing.add(key)
        added += 1

with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Added {added} examples to {path}")
print(f"Total rows: {len(rows)}")
