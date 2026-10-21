
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
    intent = intent_encoder.inverse_transform([intent_index])[0]
    intent_confidence = float(intent_probabilities[intent_index])

    # Predict urgency using rules and ML fallback
    urgency = predict_rule_urgency(message, intent, urgency_model, urgency_encoder)

    # Keep original ML urgency and confidence for comparison
    urgency_probabilities = urgency_model.predict_proba([message])[0]
    urgency_index = urgency_probabilities.argmax()
    ml_urgency = urgency_encoder.inverse_transform([urgency_index])[0]
    urgency_confidence = float(urgency_probabilities[urgency_index])

    ambiguous_delivery = is_ambiguous_delivery(message, intent)

    # Review if confidence is low, urgency is high,
    # or delivery urgency is ambiguous
    escalate = (
        intent_confidence < 0.60
        or urgency_confidence < 0.60
        or urgency == "high"
        or ambiguous_delivery
    )

    return {
        "message": message,
        "intent": intent,
        "intent_confidence": round(intent_confidence, 3),
        "urgency": urgency,
        "urgency_confidence": round(urgency_confidence, 3),
        "ml_urgency": ml_urgency,
        "human_review": bool(escalate),
    }


if __name__ == "__main__":
    examples = [
        "Where is my delivery",
        "My parcel is 3 days late",
        "My order has not arrived yet",
        "En shipment romba late aagudhu",
        "Bro en order innum deliver aagala",
        "Payment rendu thadava debit aagiduchu",
        "Payment failed but money was deducted",
        "My account access is blocked",
        "Password reset panna help venum",
        "En refund process status enna",
    ]

    for message in examples:
        print("\n", predict_ticket(message))
