"""Train intent and urgency models from labelled support examples.

Sources:
- data/support_tickets.csv: original manually labelled examples
- data/codemixed_expanded.csv: team-written synthetic Tanglish/Hinglish/English examples
- data/bitext_mapped.csv: optional output of prepare_bitext_data.py

The Bitext urgency labels are heuristic weak labels, not human annotations.
"""
from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

DATA_DIR = Path("data")
sources = [
    DATA_DIR / "support_tickets.csv",
    DATA_DIR / "codemixed_expanded.csv",
    DATA_DIR / "bitext_mapped.csv",
]
frames = []
for path in sources:
    if path.exists():
        frame = pd.read_csv(path)
        required = {"message", "intent", "urgency"}
        if required.issubset(frame.columns):
            frames.append(frame[["message", "intent", "urgency"]])
            print(f"Loaded {len(frame):,} rows from {path}")
        else:
            print(f"Skipping {path}: expected columns {sorted(required)}")

if not frames:
    raise SystemExit("No training data found in data/.")

data = pd.concat(frames, ignore_index=True)
data = data.dropna(subset=["message", "intent", "urgency"])
data["message"] = data["message"].astype(str).str.strip()
data["intent"] = data["intent"].astype(str).str.strip()
data["urgency"] = data["urgency"].astype(str).str.strip().str.lower()
data = data[data["message"].str.len() > 1]
data = data.drop_duplicates(subset=["message", "intent"])
if data["intent"].nunique() < 2:
    raise SystemExit("Training requires at least two intent categories.")

print("\nTraining rows:", len(data))
print("Intent distribution:\n", data["intent"].value_counts().to_string())
print("Urgency distribution:\n", data["urgency"].value_counts().to_string())

X = data["message"]
intent_encoder = LabelEncoder()
urgency_encoder = LabelEncoder()
y_intent = intent_encoder.fit_transform(data["intent"])
y_urgency = urgency_encoder.fit_transform(data["urgency"])

def make_model(class_weight=None):
    # Word n-grams capture phrases; character n-grams help with spelling variation
    # and romanized Tanglish/Hinglish tokens.
    features = [
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), strip_accents="unicode",
            sublinear_tf=True, min_df=1, max_features=120000
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode",
            sublinear_tf=True, min_df=1, max_features=160000
        )),
    ]
    from sklearn.pipeline import FeatureUnion
    classifier = CalibratedClassifierCV(
        LinearSVC(class_weight=class_weight), cv=3
    )
    return Pipeline([("features", FeatureUnion(features)), ("classifier", classifier)])

intent_model = make_model()
urgency_model = make_model(class_weight="balanced")
intent_model.fit(X, y_intent)
urgency_model.fit(X, y_urgency)

joblib.dump(intent_model, "intent_model.joblib")
joblib.dump(urgency_model, "urgency_model.joblib")
joblib.dump(intent_encoder, "intent_encoder.joblib")
joblib.dump(urgency_encoder, "urgency_encoder.joblib")

print("\nTraining completed successfully!")
print("Intent categories:", list(intent_encoder.classes_))
print("Urgency categories:", list(urgency_encoder.classes_))
print("Saved four .joblib model files in the project root.")
