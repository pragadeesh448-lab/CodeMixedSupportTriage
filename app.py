
import streamlit as st
import joblib
import re

st.set_page_config(
    page_title="Code-Mixed Support Triage",
    page_icon="🎧",
    layout="wide"
)

# Load trained models and encoders
intent_model = joblib.load("intent_model.joblib")
urgency_model = joblib.load("urgency_model.joblib")
intent_encoder = joblib.load("intent_encoder.joblib")
urgency_encoder = joblib.load("urgency_encoder.joblib")


def predict_rule_urgency(message, predicted_intent):
    text = message.lower()
    text = re.sub(r"\s+", " ", text).strip()

    # High urgency: serious payment issues
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

    # Low urgency: routine enquiries and account help
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

    # Routine return, cancellation, and refund requests
    if predicted_intent in [
        "return_request",
        "order_cancellation",
        "refund_request",
    ]:
        return "medium"

    # Use ML estimate for other messages
    probabilities = urgency_model.predict_proba([message])[0]
    index = probabilities.argmax()
    return urgency_encoder.inverse_transform([index])[0]


def get_suggested_reply(intent):
    if intent == "refund_request":
        return (
            "We understand that you are waiting for your refund. "
            "Our support team will review the transaction and "
            "provide an update. Thank you for your patience."
        )

    elif intent == "delivery_issue":
        return (
            "We are sorry for the delivery delay. "
            "Our support team will check your order status "
            "and help you with the next steps."
        )

    elif intent == "payment_issue":
        return (
            "We understand your payment concern. "
            "Please do not make another payment while we review "
            "the transaction. Our support team will investigate."
        )

    elif intent == "account_support":
        return (
            "We are sorry you are having trouble with your account. "
            "Our support team will help you resolve the issue."
        )

    elif intent == "order_cancellation":
        return (
            "We received your order cancellation request. "
            "Our support team will check the order status "
            "and confirm the next steps."
        )

    elif intent == "return_request":
        return (
            "We understand you want to return your product. "
            "Our support team will review the return request "
            "and guide you through the next steps."
        )

    return (
        "Thank you for contacting support. "
        "Our team will review your request and assist you."
    )


st.title("Code-Mixed Support Triage Assistant")
st.write(
    "Classify customer messages written in Tanglish, "
    "Hinglish, or English."
)

message = st.text_area(
    "Enter customer support message",
    placeholder="Example: Bro en order innum deliver aagala"
)

if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a customer message.")

    else:
        # Predict intent
        intent_probs = intent_model.predict_proba([message])[0]
        intent_index = intent_probs.argmax()
        intent = intent_encoder.inverse_transform([intent_index])[0]
        intent_conf = float(intent_probs[intent_index])

        # Keep original ML urgency prediction for transparency
        urgency_probs = urgency_model.predict_proba([message])[0]
        urgency_index = urgency_probs.argmax()
        ml_urgency = urgency_encoder.inverse_transform([urgency_index])[0]
        urgency_conf = float(urgency_probs[urgency_index])

        # Apply rules to produce the final urgency estimate
        urgency = predict_rule_urgency(message, intent)

        text = message.lower()

        # Ambiguous delivery messages need human review
        explicit_delay = any(term in text for term in [
            "3 days",
            "three days",
            "teen din",
            "romba late",
            "bahut late",
            "romba naal",
        ])

        ambiguous_delivery = (
            intent == "delivery_issue"
            and not explicit_delay
            and urgency != "high"
        )

        human_review = (
            intent_conf < 0.60
            or urgency_conf < 0.60
            or urgency == "high"
            or ambiguous_delivery
        )

        st.subheader("Triage Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Intent", intent)
            st.metric("Intent Confidence", f"{intent_conf:.1%}")

        with col2:
            st.metric("Predicted Urgency", urgency)
            st.metric("ML Urgency Estimate", ml_urgency)
            st.metric("ML Urgency Confidence", f"{urgency_conf:.1%}")

        if human_review:
            st.warning("Human review recommended")
        else:
            st.success(
                "No review flag triggered by the current prototype rules"
            )

        st.subheader("Suggested Customer Reply")

        reply = get_suggested_reply(intent)

        st.text_area(
            "Draft reply (review before sending)",
            value=reply,
            height=120
        )

        st.caption(
            "Prototype only: urgency rules and confidence scores are "
            "not validated reliability measures. Review urgent and "
            "ambiguous messages carefully."
        )
