
import streamlit as st
import joblib

st.set_page_config(
    page_title="Code-Mixed Support Triage",
    page_icon="🎧",
    layout="wide"
)

intent_model = joblib.load("intent_model.joblib")
urgency_model = joblib.load("urgency_model.joblib")
intent_encoder = joblib.load("intent_encoder.joblib")
urgency_encoder = joblib.load("urgency_encoder.joblib")

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
        intent_probs = intent_model.predict_proba([message])[0]
        urgency_probs = urgency_model.predict_proba([message])[0]

        intent_index = intent_probs.argmax()
        urgency_index = urgency_probs.argmax()

        intent = intent_encoder.inverse_transform([intent_index])[0]
        urgency = urgency_encoder.inverse_transform([urgency_index])[0]

        intent_conf = float(intent_probs[intent_index])
        urgency_conf = float(urgency_probs[urgency_index])

        human_review = (
            intent_conf < 0.60
            or urgency_conf < 0.60
            or urgency == "high"
        )

        st.subheader("Triage Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Intent", intent)
            st.metric("Intent Confidence", f"{intent_conf:.1%}")

        with col2:
            st.metric("Predicted Urgency", urgency)
            st.metric("Urgency Confidence", f"{urgency_conf:.1%}")

        if human_review:
            st.warning("Human review recommended")
        else:
            st.success("No review flag triggered by the current prototype rules")

        st.caption(
            "Prototype only: confidence scores are not validated "
            "reliability measures. Review urgent messages carefully."
        )
