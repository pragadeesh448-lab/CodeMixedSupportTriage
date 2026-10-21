
import streamlit as st
import joblib
from triage_rules import predict_rule_urgency, is_ambiguous_delivery

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
        urgency = predict_rule_urgency(message, intent, urgency_model, urgency_encoder)

        ambiguous_delivery = is_ambiguous_delivery(message, intent) and urgency != "high"

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
