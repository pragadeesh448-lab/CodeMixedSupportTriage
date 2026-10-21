
import joblib

# Load the trained models and label encoders
intent_model = joblib.load("intent_model.joblib")
urgency_model = joblib.load("urgency_model.joblib")

intent_encoder = joblib.load("intent_encoder.joblib")
urgency_encoder = joblib.load("urgency_encoder.joblib")


def predict_ticket(message):
    # Predict intent
    intent_probabilities = intent_model.predict_proba([message])[0]
    intent_index = intent_probabilities.argmax()
    intent = intent_encoder.inverse_transform([intent_index])[0]
    intent_confidence = intent_probabilities[intent_index]

    # Predict urgency
    urgency_probabilities = urgency_model.predict_proba([message])[0]
    urgency_index = urgency_probabilities.argmax()
    urgency = urgency_encoder.inverse_transform([urgency_index])[0]
    urgency_confidence = urgency_probabilities[urgency_index]

    # Conservative escalation rule for the prototype
    escalate = (
        intent_confidence < 0.60
        or urgency_confidence < 0.60
        or urgency == "high"
    )

    return {
        "message": message,
        "intent": intent,
        "intent_confidence": round(float(intent_confidence), 3),
        "urgency": urgency,
        "urgency_confidence": round(float(urgency_confidence), 3),
        "human_review": bool(escalate),
    }


if __name__ == "__main__":
    examples = [
        "Bro en order innum deliver aagala",
        "Payment rendu thadava debit aagiduchu",
        "Refund approve aachu but amount credit aagala",
        "En account login aagala",
        "Mera order abhi tak deliver nahi hua",
        "Mere account se do baar paise kat gaye",
        "Mera refund abhi tak nahi aaya",
        "Mera account login nahi ho raha",
        "My order has not arrived yet",
        "My payment was deducted twice",
    ]

    for message in examples:
        result = predict_ticket(message)
        print("\n", result)
