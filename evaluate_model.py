import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

from triage_rules import predict_rule_urgency, is_ambiguous_delivery

# Load trained models and label encoders.
intent_model = joblib.load("intent_model.joblib")
urgency_model = joblib.load("urgency_model.joblib")
intent_encoder = joblib.load("intent_encoder.joblib")
urgency_encoder = joblib.load("urgency_encoder.joblib")

# This is a separate, manually labelled hold-out dataset.
data = pd.read_csv("data/test_cases.csv").dropna(
    subset=["message", "expected_intent", "expected_urgency"]
)
messages = data["message"].astype(str)

# Predict intent and preserve the ML-only urgency estimate.
intent_predictions = intent_model.predict(messages)
urgency_predictions = urgency_model.predict(messages)
predicted_intents = intent_encoder.inverse_transform(intent_predictions)
ml_urgencies = urgency_encoder.inverse_transform(urgency_predictions)

# Apply the same rule layer used by app.py and test_model.py.
rule_urgencies = [
    predict_rule_urgency(message, intent, urgency_model, urgency_encoder)
    for message, intent in zip(messages, predicted_intents)
]

results = data.copy()
results["predicted_intent"] = predicted_intents
results["ml_urgency"] = ml_urgencies
results["rule_assisted_urgency"] = rule_urgencies
results["intent_correct"] = results["expected_intent"] == results["predicted_intent"]
results["ml_urgency_correct"] = results["expected_urgency"] == results["ml_urgency"]
results["rule_assisted_urgency_correct"] = (
    results["expected_urgency"] == results["rule_assisted_urgency"]
)
results["ambiguous_delivery_review"] = [
    is_ambiguous_delivery(message, intent)
    for message, intent in zip(messages, predicted_intents)
]

def print_accuracy(label, expected, predicted):
    score = accuracy_score(expected, predicted) * 100
    print(f"{label}: {score:.2f}%")

print("Unseen test messages:", len(results))
print_accuracy("Intent accuracy", results["expected_intent"], results["predicted_intent"])
print_accuracy("ML-only urgency accuracy", results["expected_urgency"], results["ml_urgency"])
print_accuracy("Rule-assisted urgency accuracy", results["expected_urgency"], results["rule_assisted_urgency"])

print("\nIntent classification report:")
print(classification_report(
    results["expected_intent"], results["predicted_intent"], zero_division=0
))

print("\nML-only urgency classification report:")
print(classification_report(
    results["expected_urgency"], results["ml_urgency"], zero_division=0
))

print("\nRule-assisted urgency classification report:")
print(classification_report(
    results["expected_urgency"], results["rule_assisted_urgency"], zero_division=0
))

print("\nIncorrect intent predictions:")
print(results.loc[
    ~results["intent_correct"],
    ["message", "expected_intent", "predicted_intent"]
].to_string(index=False))

print("\nIncorrect ML-only urgency predictions:")
print(results.loc[
    ~results["ml_urgency_correct"],
    ["message", "expected_urgency", "ml_urgency"]
].to_string(index=False))

print("\nIncorrect rule-assisted urgency predictions:")
print(results.loc[
    ~results["rule_assisted_urgency_correct"],
    ["message", "expected_urgency", "rule_assisted_urgency"]
].to_string(index=False))

results.to_csv("evaluation_results.csv", index=False)
print("\nDetailed results saved to evaluation_results.csv")
