
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load trained models
intent_model = joblib.load("intent_model.joblib")
urgency_model = joblib.load("urgency_model.joblib")

intent_encoder = joblib.load("intent_encoder.joblib")
urgency_encoder = joblib.load("urgency_encoder.joblib")

# Load separate test dataset
data = pd.read_csv("data/test_cases.csv").dropna(
    subset=["message", "expected_intent", "expected_urgency"]
)

messages = data["message"].astype(str)

# Predict intent and urgency
intent_predictions = intent_model.predict(messages)
urgency_predictions = urgency_model.predict(messages)

predicted_intents = intent_encoder.inverse_transform(intent_predictions)
predicted_urgencies = urgency_encoder.inverse_transform(urgency_predictions)

# Compare expected and predicted labels
results = data.copy()
results["predicted_intent"] = predicted_intents
results["predicted_urgency"] = predicted_urgencies

results["intent_correct"] = (
    results["expected_intent"] == results["predicted_intent"]
)
results["urgency_correct"] = (
    results["expected_urgency"] == results["predicted_urgency"]
)

# Print accuracy
print("Unseen test messages:", len(results))
print(
    "Intent accuracy:",
    round(accuracy_score(
        results["expected_intent"], predicted_intents
    ) * 100, 2), "%"
)
print(
    "Urgency accuracy:",
    round(accuracy_score(
        results["expected_urgency"], predicted_urgencies
    ) * 100, 2), "%"
)

print("\nIntent classification report:")
print(classification_report(
    results["expected_intent"],
    results["predicted_intent"],
    zero_division=0
))

print("\nUrgency classification report:")
print(classification_report(
    results["expected_urgency"],
    results["predicted_urgency"],
    zero_division=0
))

print("\nIncorrect intent predictions:")
print(results.loc[
    ~results["intent_correct"],
    ["message", "expected_intent", "predicted_intent"]
].to_string(index=False))

# Save detailed results
results.to_csv("evaluation_results.csv", index=False)
print("\nResults saved to evaluation_results.csv")
