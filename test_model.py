
import joblib
import re

# Load trained models and label encoders
intent_model = joblib.load("intent_model.joblib")
urgency_model = joblib.load("urgency_model.joblib")

intent_encoder = joblib.load("intent_encoder.joblib")
urgency_encoder = joblib.load("urgency_encoder.joblib")


def predict_rule_urgency(message, predicted_intent):
    text = message.lower()
    text = re.sub(r"\s+", " ", text).strip()

    # High urgency: serious payment problems
    high_payment_patterns = [
        "deducted twice",
        "charged twice",
        "double debit",
        "payment failed but",
        "payment failed and",
        "money was deducted",
        "amount was deducted",
        "paise kat gaye",
        "paise kat gaye lekin",
        "rendu thadava debit",
        "extra money deduct",
        "extra amount cut",
        "amount debit aagiduchu",
        "money was deducted",
        "amount was debited",
    ]

    # High urgency: explicit delays
    high_delay_patterns = [
        "3 days late",
        "3 days ah varala",
        "three days late",
        "teen din se nahi aaya",
        "romba late",
        "bahut late",
        "romba naal aachu",
        "refund innum varala",
        "refund abhi tak nahi aaya",
        "refund nahi mila",
        "refund amount urgent",
        "refund urgent",
    ]

    # High urgency: blocked accounts
    high_account_patterns = [
        "account locked",
        "account lock",
        "account is blocked",
        "access is blocked",
        "account lock ho gaya",
        "account access is blocked",
    ]

    if any(pattern in text for pattern in high_payment_patterns):
        return "high"

    if any(pattern in text for pattern in high_delay_patterns):
        return "high"

    if any(pattern in text for pattern in high_account_patterns):
        return "high"

    # Low urgency: routine enquiries
    low_patterns = [
        "password reset",
        "password marandhuten",
        "password bhool gaya",
        "forgot my password",
        "forgot password",
        "help me sign in",
        "help me login",
        "sign in to my account",
        "refund status",
        "refund process status",
        "refund status check",
        "status check",
        "status update",
        "where is my delivery",
        "order tracking status",
        "order ka track",
    ]

    if any(pattern in text for pattern in low_patterns):
        return "low"

    # Normal return or cancellation requests
    if predicted_intent in ["return_request", "order_cancellation"]:
        return "medium"

    # A normal refund request is medium unless an explicit high/low rule matched
    if predicted_intent == "refund_request":
        return "medium"

    # Delivery messages without a clear delay need human review.
    # Keep the ML prediction as a provisional urgency estimate.
    if predicted_intent == "delivery_issue":
        probabilities = urgency_model.predict_proba([message])[0]
        index = probabilities.argmax()
        return urgency_encoder.inverse_transform([index])[0]

    # ML fallback for other messages
    probabilities = urgency_model.predict_proba([message])[0]
    index = probabilities.argmax()
    return urgency_encoder.inverse_transform([index])[0]


def predict_ticket(message):
    # Predict intent
    intent_probabilities = intent_model.predict_proba([message])[0]
    intent_index = intent_probabilities.argmax()
    intent = intent_encoder.inverse_transform([intent_index])[0]
    intent_confidence = float(intent_probabilities[intent_index])

    # Predict urgency using rules and ML fallback
    urgency = predict_rule_urgency(message, intent)

    # Keep original ML urgency and confidence for comparison
    urgency_probabilities = urgency_model.predict_proba([message])[0]
    urgency_index = urgency_probabilities.argmax()
    ml_urgency = urgency_encoder.inverse_transform([urgency_index])[0]
    urgency_confidence = float(urgency_probabilities[urgency_index])

    text = message.lower()

    # Ambiguous delivery messages require human review
    ambiguous_delivery = (
        intent == "delivery_issue"
        and not any(term in text for term in [
            "3 days", "three days", "teen din",
            "romba late", "bahut late", "romba naal",
        ])
    )

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
