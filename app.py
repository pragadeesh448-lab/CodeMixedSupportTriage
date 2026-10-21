import streamlit as st
import joblib
from triage_rules import predict_rule_urgency, is_ambiguous_delivery

st.set_page_config(
    page_title="SupportIQ · Code-Mixed Triage",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Premium UI theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#eef2ff; --muted:#9ba8c7; --panel:#111a2e; --line:rgba(148,163,184,.16); --blue:#7c8cff; --cyan:#4de2d0; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 12% 0%, rgba(75,93,190,.19), transparent 35%), radial-gradient(ellipse at 90% 15%, rgba(42,190,184,.10), transparent 28%), #080e1b; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(8,14,27,.72); }
.block-container { max-width:1280px; padding-top:2rem; padding-bottom:3rem; }
section[data-testid="stSidebar"] { background:#0c1425; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] * { color:#dbe5ff; }
.hero { padding:28px 30px; border:1px solid rgba(139,156,255,.25); border-radius:24px; background:linear-gradient(120deg,rgba(35,48,91,.92),rgba(16,30,48,.92) 62%,rgba(12,42,52,.85)); box-shadow:0 20px 70px rgba(0,0,0,.22); margin-bottom:24px; position:relative; overflow:hidden; }
.hero:after { content:''; position:absolute; width:210px;height:210px;right:-55px;top:-95px;border-radius:50%;background:rgba(124,140,255,.13);filter:blur(2px); }
.eyebrow { text-transform:uppercase; letter-spacing:2.2px; font-size:11px; font-weight:800; color:#9daeff; margin-bottom:10px; }
.hero h1 { font-family:'Manrope',sans-serif; font-size:clamp(27px,3.5vw,42px); line-height:1.13; letter-spacing:-1.3px; color:#f7f8ff; margin:0 0 12px 0; font-weight:800; }
.hero p { color:#b8c4e2; font-size:15px; margin:0; max-width:690px; line-height:1.7; }
.pill { display:inline-block; border:1px solid rgba(77,226,208,.3); background:rgba(77,226,208,.08); color:#7cebdd; border-radius:999px; padding:6px 11px; font-size:11px; font-weight:700; margin-top:17px; }
.section-title { font-family:'Manrope',sans-serif; font-size:20px; font-weight:800; color:#f0f3ff; margin:4px 0 4px 0; }
.section-sub { color:var(--muted); font-size:13px; margin-bottom:16px; }
.panel { background:linear-gradient(145deg,rgba(18,28,49,.95),rgba(13,22,39,.95)); border:1px solid var(--line); border-radius:20px; padding:22px; margin-bottom:16px; }
.kpi { background:linear-gradient(145deg,rgba(19,30,53,.98),rgba(14,22,40,.98)); border:1px solid var(--line); border-radius:17px; padding:18px 19px; min-height:112px; }
.kpi-label { color:#9ba8c7; font-size:12px; font-weight:600; margin-bottom:10px; }
.kpi-value { color:#f4f6ff; font-family:'Manrope',sans-serif; font-size:23px; font-weight:800; overflow-wrap:anywhere; }
.kpi-note { color:#7b8aa9; font-size:11px; margin-top:7px; }
div[data-testid="stTextArea"] textarea { background:#0a1222; color:#eef2ff; border:1px solid #2a3857; border-radius:13px; line-height:1.6; }
div[data-testid="stTextArea"] textarea:focus { border-color:#7c8cff; box-shadow:0 0 0 2px rgba(124,140,255,.15); }
.stButton > button { border:0; border-radius:12px; min-height:47px; font-weight:800; background:linear-gradient(100deg,#7c8cff,#6a78ed); color:white; box-shadow:0 8px 24px rgba(93,108,230,.24); transition:transform .15s ease,filter .15s ease; }
.stButton > button:hover { filter:brightness(1.08); transform:translateY(-1px); border:0; color:white; }
.result-card { background:linear-gradient(145deg,rgba(19,30,53,.98),rgba(13,22,39,.98)); border:1px solid var(--line); border-radius:18px; padding:19px 20px; min-height:122px; }
.result-label { color:#9ba8c7; font-size:12px; font-weight:700; margin-bottom:11px; }
.result-value { color:#f4f6ff; font-family:'Manrope',sans-serif; font-size:22px; font-weight:800; overflow-wrap:anywhere; }
.badge { display:inline-block; border-radius:999px; padding:5px 10px; font-size:11px; font-weight:800; margin-top:10px; }
.badge-blue { background:rgba(124,140,255,.13); color:#aeb9ff; border:1px solid rgba(124,140,255,.22); }
.badge-green { background:rgba(77,226,208,.10); color:#75ebdc; border:1px solid rgba(77,226,208,.22); }
.badge-red { background:rgba(255,108,130,.11); color:#ff9bab; border:1px solid rgba(255,108,130,.22); }
.note { border-left:3px solid #7c8cff; background:rgba(124,140,255,.07); border-radius:0 12px 12px 0; padding:13px 15px; color:#b8c4e2; font-size:12px; line-height:1.65; }
.footer { text-align:center; color:#667594; font-size:11px; padding:28px 0 4px; }
hr { border-color:var(--line); }
[data-testid="stMetric"] { background:rgba(15,25,44,.65); border:1px solid var(--line); padding:14px 16px; border-radius:14px; }
[data-testid="stMetricLabel"] { color:#9ba8c7; }
[data-testid="stMetricValue"] { color:#f4f6ff; }
</style>
""", unsafe_allow_html=True)

# ---------- Model loading ----------
@st.cache_resource
def load_models():
    return (
        joblib.load("intent_model.joblib"),
        joblib.load("urgency_model.joblib"),
        joblib.load("intent_encoder.joblib"),
        joblib.load("urgency_encoder.joblib"),
    )

intent_model, urgency_model, intent_encoder, urgency_encoder = load_models()


def get_suggested_reply(intent):
    replies = {
        "refund_request": "We understand that you are waiting for your refund. Our support team will review the transaction and provide an update. Thank you for your patience.",
        "delivery_issue": "We are sorry for the delivery delay. Our support team will check your order status and help you with the next steps.",
        "payment_issue": "We understand your payment concern. Please do not make another payment while we review the transaction. Our support team will investigate.",
        "account_support": "We are sorry you are having trouble with your account. Our support team will help you resolve the issue.",
        "order_cancellation": "We received your order cancellation request. Our support team will check the order status and confirm the next steps.",
        "return_request": "We understand you want to return your product. Our support team will review the return request and guide you through the next steps.",
    }
    return replies.get(intent, "Thank you for contacting support. Our team will review your request and assist you.")


def pretty_label(value):
    return str(value).replace("_", " ").title()


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎧 SupportIQ")
    st.caption("CODE-MIXED CUSTOMER INTELLIGENCE")
    st.markdown("---")
    st.markdown("### Workspace")
    st.markdown("🟣 **Message triage**")
    st.markdown("📊 Intent & urgency analysis")
    st.markdown("✉️ Suggested response")
    st.markdown("---")
    st.markdown("### Language support")
    st.markdown("Tanglish · Hinglish · English")
    st.markdown("---")
    st.markdown('<div class="note">AI-assisted triage prototype. Review urgent, unclear, or low-confidence cases before taking action.</div>', unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">CUSTOMER OPERATIONS · AI WORKSPACE</div>
  <h1>Understand every message.<br>Prioritize what matters.</h1>
  <p>Turn code-mixed customer messages into actionable support insights. Detect intent, estimate urgency, and prepare a reply — all in one workspace.</p>
  <span class="pill">● &nbsp; TRIAGE ENGINE READY</span>
</div>
""", unsafe_allow_html=True)

# Compact feature cards
k1, k2, k3 = st.columns(3)
for col, icon, title, subtitle in [
    (k1, "◈", "Intent detection", "Understand what the customer needs"),
    (k2, "↗", "Urgency estimation", "Surface messages needing attention"),
    (k3, "✉", "Reply assistance", "Create a draft for agent review"),
]:
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{icon} &nbsp; CAPABILITY</div><div class="kpi-value" style="font-size:18px">{title}</div><div class="kpi-note">{subtitle}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns([1.12, 0.88], gap="large")

with left:
    st.markdown('<div class="section-title">Analyze a customer message</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Paste the message as the customer wrote it — including mixed-language words.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        message = st.text_area(
            "Customer message",
            placeholder="Example: Bro en order innum deliver aagala...",
            height=175,
            label_visibility="visible",
            key="customer_message",
        )
        sample_cols = st.columns([1, 1, 1])
        with sample_cols[0]:
            if st.button("Try Tanglish sample", use_container_width=True):
                st.session_state["customer_message"] = "Bro en order innum deliver aagala"
                st.rerun()
        with sample_cols[1]:
            if st.button("Try refund sample", use_container_width=True):
                st.session_state["customer_message"] = "My refund innum varala, please help"
                st.rerun()
        with sample_cols[2]:
            if st.button("Try payment sample", use_container_width=True):
                st.session_state["customer_message"] = "Amount debit aachu but order confirm aagala"
                st.rerun()
        analyze = st.button("✦  Analyze message", type="primary", use_container_width=True)

with right:
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A quick view of the triage workflow.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
      <div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:19px">
        <div style="background:rgba(124,140,255,.15);color:#aeb9ff;border-radius:12px;padding:10px 13px;font-weight:800">01</div>
        <div><div style="font-weight:800;color:#edf1ff;margin-bottom:4px">Read the message</div><div style="font-size:12px;color:#9ba8c7;line-height:1.6">Accept a customer message in Tanglish, Hinglish, or English.</div></div>
      </div>
      <div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:19px">
        <div style="background:rgba(77,226,208,.12);color:#75ebdc;border-radius:12px;padding:10px 13px;font-weight:800">02</div>
        <div><div style="font-weight:800;color:#edf1ff;margin-bottom:4px">Classify & prioritize</div><div style="font-size:12px;color:#9ba8c7;line-height:1.6">The models estimate intent and urgency; rule checks help flag cases for review.</div></div>
      </div>
      <div style="display:flex;gap:14px;align-items:flex-start">
        <div style="background:rgba(255,190,92,.12);color:#ffd28a;border-radius:12px;padding:10px 13px;font-weight:800">03</div>
        <div><div style="font-weight:800;color:#edf1ff;margin-bottom:4px">Review the next step</div><div style="font-size:12px;color:#9ba8c7;line-height:1.6">Use the suggested reply as a draft — not as an automatic message.</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- Analysis output ----------
if analyze:
    if not message.strip():
        st.warning("Please enter a customer message before analyzing.")
    else:
        intent_probs = intent_model.predict_proba([message])[0]
        intent_index = intent_probs.argmax()
        intent = intent_encoder.inverse_transform([intent_index])[0]
        intent_conf = float(intent_probs[intent_index])

        urgency_probs = urgency_model.predict_proba([message])[0]
        urgency_index = urgency_probs.argmax()
        ml_urgency = urgency_encoder.inverse_transform([urgency_index])[0]
        urgency_conf = float(urgency_probs[urgency_index])

        urgency = predict_rule_urgency(message, intent, urgency_model, urgency_encoder)
        ambiguous_delivery = is_ambiguous_delivery(message, intent) and urgency != "high"
        human_review = (
            intent_conf < 0.60
            or urgency_conf < 0.60
            or urgency == "high"
            or ambiguous_delivery
        )

        st.markdown("---")
        st.markdown('<div class="section-title">Triage results</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Predictions for this message. Confidence is a model score, not a guarantee of correctness.</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="result-card"><div class="result-label">DETECTED INTENT</div><div class="result-value">{pretty_label(intent)}</div><span class="badge badge-blue">{intent_conf:.0%} model score</span></div>', unsafe_allow_html=True)
        urgency_class = "badge-red" if urgency == "high" else ("badge-green" if urgency == "low" else "badge-blue")
        c2.markdown(f'<div class="result-card"><div class="result-label">FINAL URGENCY ESTIMATE</div><div class="result-value">{pretty_label(urgency)}</div><span class="badge {urgency_class}">Rule-assisted estimate</span></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="result-card"><div class="result-label">ML-ONLY URGENCY</div><div class="result-value">{pretty_label(ml_urgency)}</div><span class="badge badge-blue">{urgency_conf:.0%} model score</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if human_review:
            st.warning("Human review recommended — this message may be urgent, ambiguous, or have a lower model score.")
        else:
            st.info("No review flag was triggered by the current prototype rules. An agent should still verify the result.")

        st.markdown('<div class="section-title">Suggested customer reply</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Review and edit this draft before sending it to a customer.</div>', unsafe_allow_html=True)
        reply = get_suggested_reply(intent)
        st.text_area("Reply draft", value=reply, height=135, key="suggested_reply")
        st.caption("Prototype limitation: urgency rules and confidence scores are not validated reliability measures. Always review urgent and ambiguous messages carefully.")

st.markdown('<div class="footer">SUPPORTIQ · CODE-MIXED SUPPORT TRIAGE ASSISTANT &nbsp; • &nbsp; HUMAN-IN-THE-LOOP BY DESIGN</div>', unsafe_allow_html=True)
