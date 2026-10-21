
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV


# Load the support ticket dataset
data = pd.read_csv("data/support_tickets.csv")
data = data.dropna(subset=["message", "intent", "urgency"])

# Prepare text and labels
X = data["message"].astype(str)

intent_encoder = LabelEncoder()
urgency_encoder = LabelEncoder()

y_intent = intent_encoder.fit_transform(data["intent"])
y_urgency = urgency_encoder.fit_transform(data["urgency"])


# Build intent classification model
intent_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents="unicode"
        )
    ),
    (
        "classifier",
        CalibratedClassifierCV(
            LinearSVC(),
            cv=2
        )
    )
])


# Build urgency classification model
urgency_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents="unicode"
        )
    ),
    (
        "classifier",
        CalibratedClassifierCV(
            LinearSVC(class_weight="balanced"),
            cv=2
        )
    )
])


# Train both models
intent_model.fit(X, y_intent)
urgency_model.fit(X, y_urgency)


# Save trained models
joblib.dump(intent_model, "intent_model.joblib")
joblib.dump(urgency_model, "urgency_model.joblib")
joblib.dump(intent_encoder, "intent_encoder.joblib")
joblib.dump(urgency_encoder, "urgency_encoder.joblib")


print("Training completed successfully!")
print("Training messages:", len(data))
print("Intent categories:", list(intent_encoder.classes_))
print("Urgency categories:", list(urgency_encoder.classes_))
