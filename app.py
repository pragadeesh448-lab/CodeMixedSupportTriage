import streamlit as st
import joblib
import re
from triage_rules import predict_rule_urgency, is_ambiguous_delivery


import csv
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path("data") / "triage_history.csv"
HISTORY_FIELDS = [
    "timestamp", "message", "intent", "urgency", "intent_confidence",
    "urgency_confidence", "human_review", "review_reason", "language_style"
]

# Small labelled example bank for similarity-based context (not the held-out evaluation set).
FEW_SHOT_EXAMPLES = [
    ("charger wire miss aagudhu", "product_complaint"),
    ("box damaged ah vandhurukku", "product_complaint"),
    ("wrong product vandhurukku", "product_complaint"),
    ("product exchange pannanum", "product_exchange"),
    ("indha item exchange pannunga", "product_exchange"),
    ("enakku refund venum", "refund_request"),
    ("paise wapas chahiye", "refund_request"),
    ("product return pannanum", "return_request"),
    ("order innum deliver aagala", "delivery_issue"),
    ("amount debit aachu but order confirm aagala", "payment_issue"),
    ("login panna mudiyala", "account_support"),
    ("order cancel pannanum", "order_cancellation"),
    ("indha phone price enna", "product_inquiry"),
    ("mujhe product ki details chahiye", "product_inquiry"),
    ("thank you, product romba nalla irukku", "positive_feedback"),
]


def retrieve_similar_examples(message, top_k=3):
    """Simple word TF-IDF cosine similarity; falls back to token overlap if sklearn is unavailable."""
    examples = FEW_SHOT_EXAMPLES
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        corpus = [message] + [item[0] for item in examples]
        matrix = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode").fit_transform(corpus)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        ranked = sorted(zip(scores, examples), key=lambda pair: pair[0], reverse=True)
        return [(phrase, label, float(score)) for score, (phrase, label) in ranked[:top_k]]
    except Exception:
        tokens = set(re.findall(r"\w+", message.lower()))
        ranked = []
        for phrase, label in examples:
            other = set(re.findall(r"\w+", phrase.lower()))
            score = len(tokens & other) / max(1, len(tokens | other))
            ranked.append((phrase, label, score))
        ranked.sort(key=lambda item: item[2], reverse=True)
        return ranked[:top_k]


def save_ticket(message, intent, urgency, intent_confidence, urgency_confidence,
                human_review, review_reasons, language_style):
    """Append one analyzed ticket to a local CSV history for the queue dashboard."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": message,
            "intent": intent,
            "urgency": urgency,
            "intent_confidence": round(float(intent_confidence), 4),
            "urgency_confidence": round(float(urgency_confidence), 4),
            "human_review": bool(human_review),
            "review_reason": " ".join(review_reasons),
            "language_style": language_style,
        })


def load_ticket_history():
    if not HISTORY_PATH.exists():
        return []
    try:
        with HISTORY_PATH.open("r", newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))
    except (OSError, csv.Error):
        return []


# Explicit intent rules for clear customer actions. The ML model remains the fallback.
# These rules separate exchange, return, refund, inquiry, and complaint requests.
INTENT_RULES = [
    (
        "product_exchange",
        [
            r"\b(exchange|replace|replacement)\b",
            r"\b(exchange|replace)\s+(it|this|the product|the item|my product|my item)\b",
            r"\b(product|item|dress|shoe|phone).{0,35}\b(exchange|replace)\b",
            r"\b(change|swap)\s+(this|it|the product|the size)\b",
            r"\b(exchange|replace)\s+ pannanum\b",
            r"\bexchange pannanum\b",
            r"\breplace pannanum\b",
            r"\bsize change pannanum\b",
            r"\bexchange karna hai\b",
            r"\breplace karna hai\b",
        ],
    ),
    (
        "refund_request",
        [
            r"\b(refund|money back|give my money back)\b",
            r"\brefund venum\b",
            r"\brefund kudu\b",
            r"\brefund pannunga\b",
            r"\bpaise wapas\b",
            r"\bmujhe refund chahiye\b",
            r"\brefund chahiye\b",
        ],
    ),
    (
        "return_request",
        [
            r"\b(return|send it back|send this back)\b",
            r"\breturn pannanum\b",
            r"\breturn pannunga\b",
            r"\bwapas karna hai\b",
            r"\bitem wapas\b",
        ],
    ),
    (
        "product_inquiry",
        [
            r"\b(product details|product information|specifications|features of)\b",
            r"\b(available|availability|in stock|price of|cost of)\b.{0,40}\b(product|item|phone|laptop|shoe|dress)\b",
            r"\b(product|item|phone|laptop|shoe|dress)\b.{0,40}\b(price|cost|details|features|available|availability)\b",
            r"\bdetails sollunga\b",
            r"\bprice enna\b",
            r"\bavailable ah\b",
            r"\bkitne ka hai\b",
            r"\bdetails batao\b",
        ],
    ),
    (
        "product_complaint",
        [
            r"\b(damaged|broken|defective|faulty|wrong item|wrong product|poor quality)\b",
            # Missing product or accessories, including code-mixed phrases
            r"\b(product|item|order)\s+(kaanom|kaanom da|missing|missing ah irukku|varala)\b",
            r"\b(product|item)\s+(nalla illa|nalla illai|sari illa|seri illa|not good|is bad|is not good)\b",
            r"\b(product|item).{0,30}\b(kaanom|kanom|missing|miss aagudhu|miss aachu|varala)\b",
            r"\b(charger|charger wire|charging cable|cable|wire|accessory|accessories|parts|component|components)\b.{0,35}\b(missing|missed|not included|not provided|left out|varala|miss aagudhu)\b",
            r"\b(missing|missed|not included|not provided|left out)\b.{0,35}\b(charger|charger wire|charging cable|cable|wire|accessory|accessories|parts|component|components)\b",
            r"\bproduct nalla illa da\b",
            r"\bproduct kaanom da\b",
            r"\bproduct missing ah irukku\b",

            # Missing accessories or incomplete product package
            r"\b(missing|missed|not included|not provided|left out)\b.{0,45}\b(charger|charger wire|charging cable|cable|wire|accessory|accessories|parts|component|components)\b",
            r"\b(charger|charger wire|charging cable|cable|wire|accessory|accessories|parts|component|components)\b.{0,45}\b(missing|missed|not included|not provided|left out)\b",
            r"\b(product|package|box|parcel|order).{0,35}\b(incomplete|items missing|parts missing)\b",
            r"\b(charger|charger wire|charging cable|wire|cable) miss aagudhu\b",
            r"\b(charger|charger wire|charging cable|wire|cable) missing ah irukku\b",
            r"\b(charger|charger wire|charging cable|wire|cable) varala\b",
            r"\b(accessory|accessories|parts) missing\b",

            r"\b(product|item).{0,35}\b(not working|damaged|broken|defective|wrong)\b",
            r"\bdamage aagiduchu\b",
            r"\bdamage ah vandhu(rukku|irukku)\b",
            r"\bwrong product vandhu(rukku|irukku)\b",
            r"\bproduct maathi vandhu(rukku|irukku)\b",
            r"\b(product|item) maathi vandhu(rukku|irukku)\b",
            r"\b(product|item) (seri illa|sari illa|correct illa|vera vandhuruku|vera vandhirukku)\b",
            r"\b(item|product) seri ill(a|ada|ai|ai da|ai nga)\b",
            r"\b(item|product) sari ill(a|ada|ai|ai da|ai nga)\b",
            r"\bmacha item seri illa(da)?\b",
            r"\bproduct work aagala\b",
            r"\bkharab product\b",
            r"\bgalat product aaya\b",
            r"\bwrong item received\b",
        ],
    ),
]


def detect_rule_intent(message):
    """Return (intent, rule_name) when a clear intent phrase matches; otherwise (None, None)."""
    text = re.sub(r"\s+", " ", str(message).lower()).strip()
    for intent_name, patterns in INTENT_RULES:
        if any(re.search(pattern, text) for pattern in patterns):
            return intent_name, "Rule override"
    return None, None

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


def detect_language_style(message):
    """Heuristic language-style detection for reply templates; verify mixed-language cases."""
    text = str(message).lower()

    # Include common romanized Tamil/Tanglish words and colloquial verb endings.
    tanglish_markers = [
        r"\b(vandhuchu|vandhuch|vandhurukku|vandhirukku|varala|venum|pannanum|pannunga|pannu|aachu|aagudhu|aagala|illa|illai|irukku|enna|evlo|sollunga|kudu|bro|macha|enga|enakku|unga|da|la|ku|romba|damage ah|refund venum)\b"
    ]
    hinglish_markers = [
        r"\b(mujhe|chahiye|hai|hain|nahi|nahin|karna|kya|kaise|mera|meri|mere|paise|wapas|batao|kripya|milaa|mila|hua|hui|bahut|accha|kar do|karna hai)\b"
    ]

    tanglish_score = sum(bool(re.search(pattern, text)) for pattern in tanglish_markers)
    hinglish_score = sum(bool(re.search(pattern, text)) for pattern in hinglish_markers)

    if tanglish_score > hinglish_score:
        return "Tanglish"
    if hinglish_score > tanglish_score:
        return "Hinglish"
    return "English"


def get_suggested_reply(intent, message=""):
    """Return a cautious reply template matching the detected language style."""
    style = detect_language_style(message)

    replies = {
        "English": {
            "product_inquiry": "Thanks for your interest. Our support team can help confirm the product details, price, and availability.",
            "product_exchange": "We can help with your product exchange. Our support team will check the item and exchange eligibility, then guide you through the next steps.",
            "refund_request": "We understand that you are waiting for your refund. Our support team will review the transaction and provide an update. Thank you for your patience.",
            "delivery_issue": "We are sorry for the delivery delay. Our support team will check your order status and help you with the next steps.",
            "payment_issue": "We understand your payment concern. Please do not make another payment while we review the transaction. Our support team will investigate.",
            "account_support": "We are sorry you are having trouble with your account. Our support team will help you resolve the issue.",
            "order_cancellation": "We received your order cancellation request. Our support team will check the order status and confirm the next steps.",
            "return_request": "We understand you want to return your product. Our support team will review the return request and guide you through the next steps.",
            "positive_feedback": "Thank you so much for your kind feedback! We are happy to hear that you are enjoying the product.",
            "product_complaint": "We are sorry the product did not meet your expectations. Our support team will review the issue and help with the available next steps.",
        },
        "Tanglish": {
            "product_inquiry": "Indha product-oda details, price, availability ellam enga support team confirm panni solluvanga.",
            "product_exchange": "Unga exchange request-ku help pannuvom. Item-oda eligibility-ai support team check panni next steps solluvanga.",
            "refund_request": "Unga refund request-ai enga support team review panni status update pannuvanga. Konjam wait pannunga.",
            "delivery_issue": "Unga delivery delay-ku sorry. Order status-ai support team check panni next steps solluvanga.",
            "payment_issue": "Unga payment issue-ku sorry. Review mudiyura varaikkum innoru payment panna vendam. Support team transaction-ai check pannuvanga.",
            "account_support": "Unga account issue-ku sorry. Enga support team resolve panna help pannuvanga.",
            "order_cancellation": "Unga cancellation request receive pannitom. Order status check panni next steps confirm pannuvanga.",
            "return_request": "Unga return request-ai support team review panni next steps guide pannuvanga.",
            "positive_feedback": "Nalla feedback kuduthadhukku romba nandri! Neenga product-ai enjoy panreenga-nu sandhosham.",
            "product_complaint": "Product issue-ku sorry. Enga support team complaint-ai review panni available next steps-la help pannuvanga.",
        },
        "Hinglish": {
            "product_inquiry": "Is product ki details, price aur availability hamari support team confirm karke bata degi.",
            "product_exchange": "Hum aapki exchange request mein help karenge. Support team item ki eligibility check karke next steps batayegi.",
            "refund_request": "Aapki refund request hamari support team review karegi aur status update degi. Kripya thoda intezaar karein.",
            "delivery_issue": "Delivery delay ke liye humein khed hai. Support team aapke order ka status check karke next steps batayegi.",
            "payment_issue": "Payment issue ke liye humein khed hai. Review hone tak please dobara payment na karein. Support team transaction check karegi.",
            "account_support": "Account issue ke liye humein khed hai. Hamari support team aapki madad karegi.",
            "order_cancellation": "Aapki cancellation request mil gayi hai. Support team order status check karke next steps confirm karegi.",
            "return_request": "Aapki return request support team review karegi aur aapko next steps batayegi.",
            "positive_feedback": "Aapke feedback ke liye bahut shukriya! Humein khushi hai ki aapko product pasand aaya.",
            "product_complaint": "Product issue ke liye humein khed hai. Support team complaint review karke aapko available next steps batayegi.",
        },
    }

    language_replies = replies.get(style, replies["English"])
    return language_replies.get(
        intent,
        "Thank you for contacting support. Our team will review your request and assist you."
    ), style


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

    def set_sample(text):
        st.session_state["customer_message"] = text

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
            st.button(
                "Try Tanglish sample",
                use_container_width=True,
                on_click=set_sample,
                args=("Bro en order innum deliver aagala",),
            )
        with sample_cols[1]:
            st.button(
                "Try refund sample",
                use_container_width=True,
                on_click=set_sample,
                args=("My refund innum varala, please help",),
            )
        with sample_cols[2]:
            st.button(
                "Try payment sample",
                use_container_width=True,
                on_click=set_sample,
                args=("Amount debit aachu but order confirm aagala",),
            )
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
        ml_intent = intent_encoder.inverse_transform([intent_index])[0]
        ml_intent_conf = float(intent_probs[intent_index])

        rule_intent, intent_source = detect_rule_intent(message)
        if rule_intent:
            intent = rule_intent
            intent_conf = ml_intent_conf  # ML score is shown separately; it is not a confidence for the rule.
        else:
            intent = ml_intent
            intent_conf = ml_intent_conf
            intent_source = "ML model"

        urgency_probs = urgency_model.predict_proba([message])[0]
        urgency_index = urgency_probs.argmax()
        ml_urgency = urgency_encoder.inverse_transform([urgency_index])[0]
        urgency_conf = float(urgency_probs[urgency_index])

        urgency = predict_rule_urgency(message, intent, urgency_model, urgency_encoder)
        ambiguous_delivery = is_ambiguous_delivery(message, intent) and urgency != "high"

        # Explain exactly why this message is flagged for human review.
        review_reasons = []

        if intent_conf < 0.60:
            review_reasons.append("Intent prediction confidence is low.")

        if urgency_conf < 0.60:
            review_reasons.append("Urgency prediction confidence is low.")

        if urgency == "high":
            review_reasons.append("High urgency detected; human review is recommended.")

        if ambiguous_delivery:
            review_reasons.append("Delivery issue detected, but the delay duration is unclear.")

        human_review = bool(review_reasons)

        st.markdown("---")
        st.markdown('<div class="section-title">Triage results</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Predictions for this message. Confidence is a model score, not a guarantee of correctness.</div>', unsafe_allow_html=True)
        if intent_source == "Rule override":
            st.caption("Intent matched by a phrase rule. Check the wording if the label does not fit the customer's meaning.")

        c1, c2, c3 = st.columns(3)
        intent_badge = (
            '<span class="badge badge-green">Rule-based match</span>'
            if intent_source == "Rule override"
            else f'<span class="badge badge-blue">{intent_conf:.0%} model score</span>'
        )
        c1.markdown(
            f'<div class="result-card"><div class="result-label">DETECTED INTENT</div>'
            f'<div class="result-value">{pretty_label(intent)}</div>{intent_badge}</div>',
            unsafe_allow_html=True,
        )
        urgency_class = "badge-red" if urgency == "high" else ("badge-green" if urgency == "low" else "badge-blue")
        c2.markdown(f'<div class="result-card"><div class="result-label">FINAL URGENCY ESTIMATE</div><div class="result-value">{pretty_label(urgency)}</div><span class="badge {urgency_class}">Rule-assisted estimate</span></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="result-card"><div class="result-label">ML-ONLY URGENCY</div><div class="result-value">{pretty_label(ml_urgency)}</div><span class="badge badge-blue">{urgency_conf:.0%} model score</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if human_review:
            st.warning("Human review recommended.")
            for reason in review_reasons:
                st.write(f"- {reason}")
        else:
            st.info(
                "No review flag was triggered by the current prototype rules. "
                "An agent should still verify the result."
            )

        st.markdown('<div class="section-title">Suggested customer reply</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Review and edit this draft before sending it to a customer.</div>', unsafe_allow_html=True)
        reply, reply_style = get_suggested_reply(intent, message)
        st.caption(f"Reply style selected: {reply_style} (heuristic detection; please verify before sending).")
        st.text_area("Reply draft", value=reply, height=135, key="suggested_reply")

        similar_examples = retrieve_similar_examples(message)
        with st.expander("Similar labelled examples (few-shot retrieval)", expanded=True):
            st.caption("Retrieved by TF-IDF similarity from a small labelled example bank. These examples support review; they do not override the classifier.")
            for example_text, example_intent, similarity in similar_examples:
                st.write(f"- “{example_text}” → **{pretty_label(example_intent)}** · similarity {similarity:.2f}")

        # Save each submitted analysis to the local ticket history.
        if st.session_state.get("last_saved_message") != (message, intent, urgency):
            save_ticket(message, intent, urgency, intent_conf, urgency_conf,
                        human_review, review_reasons, reply_style)
            st.session_state["last_saved_message"] = (message, intent, urgency)

        st.caption("Prototype limitation: urgency rules and confidence scores are not validated reliability measures. Always review urgent and ambiguous messages carefully.")


# ---------- Persistent queue dashboard ----------
st.markdown("---")
st.markdown('<div class="section-title">Support queue dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Counts are based on messages analyzed in this local app and saved to data/triage_history.csv.</div>', unsafe_allow_html=True)
history = load_ticket_history()
if not history:
    st.info("No analyzed tickets saved yet. Analyze a customer message above to start building the queue.")
else:
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df["intent"] = history_df["intent"].fillna("unknown")
    history_df["urgency"] = history_df["urgency"].fillna("unknown")
    history_df["human_review"] = history_df["human_review"].astype(str).str.lower().isin(["true", "1", "yes"])

    total_count = len(history_df)
    high_count = int((history_df["urgency"].str.lower() == "high").sum())
    review_count = int(history_df["human_review"].sum())
    q1, q2, q3 = st.columns(3)
    q1.metric("Analyzed tickets", total_count)
    q2.metric("High urgency", high_count)
    q3.metric("Human review", review_count)

    left_chart, right_chart = st.columns(2)
    with left_chart:
        st.markdown("**Volume by intent**")
        intent_counts = history_df["intent"].value_counts().rename_axis("Intent").reset_index(name="Tickets")
        st.bar_chart(intent_counts.set_index("Intent"))
    with right_chart:
        st.markdown("**Volume by urgency**")
        urgency_counts = history_df["urgency"].value_counts().rename_axis("Urgency").reset_index(name="Tickets")
        st.bar_chart(urgency_counts.set_index("Urgency"))

    st.markdown("**Recent queue items**")
    display_cols = ["timestamp", "message", "intent", "urgency", "human_review", "review_reason"]
    st.dataframe(history_df[display_cols].tail(20).iloc[::-1], use_container_width=True, hide_index=True)
    st.download_button(
        "Download ticket history CSV",
        data=HISTORY_PATH.read_bytes(),
        file_name="supportiq_ticket_history.csv",
        mime="text/csv",
    )

st.markdown('<div class="footer">SUPPORTIQ · CODE-MIXED SUPPORT TRIAGE ASSISTANT &nbsp; • &nbsp; HUMAN-IN-THE-LOOP BY DESIGN</div>', unsafe_allow_html=True)
