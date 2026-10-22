import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import joblib
from triage_rules import predict_rule_urgency, is_ambiguous_delivery

# Load trained models and label encoders
intent_model = joblib.load("intent_model.joblib")
urgency_model = joblib.load("urgency_model.joblib")

intent_encoder = joblib.load("intent_encoder.joblib")
urgency_encoder = joblib.load("urgency_encoder.joblib")


def predict_ticket(message):
    # Predict intent
    intent_probabilities = intent_model.predict_proba([message])[0]
    intent_index = intent_probabilities.argmax()
    predicted_intent = intent_encoder.inverse_transform([intent_index])[0]
    intent_confidence = float(intent_probabilities[intent_index])

    # Preserve the ML-only urgency estimate for comparison with rule logic.
    urgency_probabilities = urgency_model.predict_proba([message])[0]
    urgency_index = urgency_probabilities.argmax()
    ml_urgency = urgency_encoder.inverse_transform([urgency_index])[0]
    ml_urgency_confidence = float(urgency_probabilities[urgency_index])

    predicted_urgency = predict_rule_urgency(message, predicted_intent, urgency_model, urgency_encoder)
    ambiguous_delivery = is_ambiguous_delivery(message, predicted_intent) and predicted_urgency != "high"

    # Explain exactly why this message is flagged for human review.
    review_reasons = []

    if intent_confidence < 0.60:
        review_reasons.append("Intent prediction confidence is low.")

    if ml_urgency_confidence < 0.60:
        review_reasons.append("Urgency prediction confidence is low.")

    if predicted_urgency == "high":
        review_reasons.append("High urgency detected; human review is recommended.")

    if ambiguous_delivery:
        review_reasons.append("Delivery issue detected, but the delay duration is unclear.")

    return {
        "message": message,
        "predicted_intent": predicted_intent,
        "intent_confidence": round(intent_confidence, 4),
        "predicted_urgency": predicted_urgency,
        "ml_urgency": ml_urgency,
        "urgency_confidence": round(ml_urgency_confidence, 4),
        "needs_human_review": bool(review_reasons),
        "human_review_reasons": review_reasons,
    }


test_messages = [
    "delivery late aagudhu, three days aachu",
    "delivery delay aagudhu, please help",
    "paise kat gaye lekin order confirm nahi hua",
    "order confirm nahi hua",
    "account is blocked, help immediately",
    "forgot my password",
]

for msg in test_messages:
    result = predict_ticket(msg)
    print("Message:", result["message"])
    print("Intent:", result["predicted_intent"], f"(conf: {result['intent_confidence']})")
    print("Urgency:", result["predicted_urgency"], f"(ML-only: {result['ml_urgency']}, conf: {result['urgency_confidence']})")
    print("Human review needed:", result["needs_human_review"])
    if result["needs_human_review"]:
        print("Review reasons:", result["human_review_reasons"])
    print("-" * 50)
