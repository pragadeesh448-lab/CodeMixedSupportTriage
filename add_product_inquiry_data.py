import csv

file_path = "data/codemixed_expanded.csv"

new_rows = [
    ("indha product oda price enna", "product_inquiry", "low"),
    ("indha item evlo cost", "product_inquiry", "low"),
    ("phone oda specifications sollunga", "product_inquiry", "low"),
    ("laptop la enna features irukku", "product_inquiry", "low"),
    ("indha product available ah", "product_inquiry", "low"),
    ("stock irukka indha model", "product_inquiry", "low"),
    ("vera colours available ah", "product_inquiry", "low"),
    ("size options enna irukku", "product_inquiry", "low"),
    ("product details share pannunga", "product_inquiry", "low"),
    ("indha phone pathi details venum", "product_inquiry", "low"),
    ("delivery charge evlo", "product_inquiry", "low"),
    ("warranty evlo years", "product_inquiry", "low"),
    ("does this product have a warranty", "product_inquiry", "low"),
    ("What is the price of this item?", "product_inquiry", "low"),
    ("Can you tell me more about this product?", "product_inquiry", "low"),
    ("Is this model currently available?", "product_inquiry", "low"),
    ("What colours does this phone come in?", "product_inquiry", "low"),
    ("Does this laptop have 16GB RAM?", "product_inquiry", "low"),
    ("What are the product specifications?", "product_inquiry", "low"),
    ("How long is the warranty period?", "product_inquiry", "low"),
    ("Is free delivery available?", "product_inquiry", "low"),
    ("mujhe is product ki price batao", "product_inquiry", "low"),
    ("ye item stock mein hai kya", "product_inquiry", "low"),
    ("is phone ke features kya hain", "product_inquiry", "low"),
    ("product ka warranty kitna hai", "product_inquiry", "low"),
    ("is product ka price kya hai", "product_inquiry", "low"),
    ("kya ye model available hai", "product_inquiry", "low"),
    ("is laptop mein kitni RAM hai", "product_inquiry", "low"),
    ("mujhe product ki details chahiye", "product_inquiry", "low"),
    ("delivery charges kitne hain", "product_inquiry", "low"),
]

with open(file_path, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames
    existing_rows = list(reader)

existing = {
    (row["message"].strip().lower(), row["intent"].strip(), row["urgency"].strip())
    for row in existing_rows
}

added = 0
for message, intent, urgency in new_rows:
    key = (message.lower(), intent, urgency)
    if key not in existing:
        existing_rows.append({
            "message": message,
            "intent": intent,
            "urgency": urgency
        })
        existing.add(key)
        added += 1

with open(file_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(existing_rows)

print(f"Added {added} new examples.")
print(f"Total rows in codemixed_expanded.csv: {len(existing_rows)}")
