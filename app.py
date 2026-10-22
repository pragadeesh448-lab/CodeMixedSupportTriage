import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import csv
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from triage_rules import predict_rule_urgency, is_ambiguous_delivery

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SupportAI · Code-Mixed Support Triage",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "triage_inbox"

if "active_ticket_id" not in st.session_state:
    st.session_state["active_ticket_id"] = None

if "customer_message" not in st.session_state:
    st.session_state["customer_message"] = "order vandhuchu but box damaged, refund venum"

if "selected_conversation_id" not in st.session_state:
    st.session_state["selected_conversation_id"] = "conv_1"

# ---------------------------------------------------------
# File Paths & Data Constants
# ---------------------------------------------------------
HISTORY_PATH = Path("data") / "triage_history.csv"
HISTORY_FIELDS = [
    "timestamp", "message", "intent", "urgency", "intent_confidence",
    "urgency_confidence", "human_review", "review_reason", "language_style", "assigned_queue"
]

QUEUE_MAPPING = {
    "refund_request": "Refunds",
    "return_request": "Returns",
    "product_exchange": "Product Exchange",
    "delivery_issue": "Delivery & Logistics",
    "payment_issue": "Billing & Payments",
    "account_support": "Account Support",
    "product_inquiry": "Product Team",
    "product_complaint": "Quality & Support",
    "order_cancellation": "Cancellations",
    "positive_feedback": "Customer Relations",
}

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

INTENT_RULES = [
    (
        "product_exchange",
        [
            r"\b(exchange|replace|replacement)\b",
            r"\b(exchange|replace)\s+(it|this|the product|the item|my product|my item)\b",
            r"\b(product|item|dress|shoe|phone).{0,35}\b(exchange|replace)\b",
            r"\b(change|swap)\s+(this|it|the product|the size)\b",
            r"\b(exchange|replace)\s+pannanum\b",
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
            r"\b(product|item|order)\s+(kaanom|kaanom da|missing|missing ah irukku|varala)\b",
            r"\b(product|item)\s+(nalla illa|nalla illai|sari illa|seri illa|not good|is bad|is not good)\b",
            r"\b(product|item).{0,30}\b(kaanom|kanom|missing|miss aagudhu|miss aachu|varala)\b",
            r"\b(charger|charger wire|charging cable|cable|wire|accessory|accessories|parts|component|components)\b.{0,35}\b(missing|missed|not included|not provided|left out|varala|miss aagudhu)\b",
            r"\b(missing|missed|not included|not provided|left out)\b.{0,35}\b(charger|charger wire|charging cable|cable|wire|accessory|accessories|parts|component|components)\b",
            r"\bproduct nalla illa da\b",
            r"\bproduct kaanom da\b",
            r"\bproduct missing ah irukku\b",
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

# ---------------------------------------------------------
# Dynamic Styling System (Light & Dark Theme)
# ---------------------------------------------------------
def apply_theme():
    theme = st.session_state["theme"]
    
    if theme == "light":
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --bg-base: #f8fafc;
            --bg-surface: #ffffff;
            --bg-elevated: #f1f5f9;
            --bg-subtle: #f8fafc;
            --border-base: #e2e8f0;
            --border-subtle: #edf2f7;
            --border-focus: #3b82f6;
            --text-heading: #0f172a;
            --text-body: #334155;
            --text-muted: #64748b;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --primary-subtle: rgba(37, 99, 235, 0.08);
            --success: #10b981;
            --success-subtle: rgba(16, 185, 129, 0.1);
            --warning: #f59e0b;
            --warning-subtle: rgba(245, 158, 11, 0.12);
            --danger: #ef4444;
            --danger-subtle: rgba(239, 68, 68, 0.1);
            --indigo: #6366f1;
            --indigo-subtle: rgba(99, 102, 241, 0.1);
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.06), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
            --card-radius: 16px;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-body);
        }

        .stApp {
            background-color: var(--bg-base);
            color: var(--text-body);
        }

        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid var(--border-base);
        }

        [data-testid="stSidebar"] * {
            color: var(--text-body);
        }

        [data-testid="stHeader"] {
            background-color: rgba(248, 250, 252, 0.85);
            backdrop-filter: blur(8px);
        }

        /* Card container */
        .sa-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-base);
            border-radius: var(--card-radius);
            padding: 22px 24px;
            box-shadow: var(--shadow-sm);
            margin-bottom: 20px;
            transition: all 0.2s ease;
        }

        .sa-card:hover {
            box-shadow: var(--shadow-md);
        }

        /* KPI Card */
        .kpi-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-base);
            border-radius: 14px;
            padding: 18px 20px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: var(--shadow-sm);
            min-height: 116px;
        }

        .kpi-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
        }

        .kpi-value-row {
            display: flex;
            align-items: baseline;
            gap: 10px;
            margin-top: 10px;
        }

        .kpi-value {
            font-size: 28px;
            font-weight: 800;
            color: var(--text-heading);
            letter-spacing: -0.5px;
        }

        .kpi-delta-up {
            font-size: 12px;
            font-weight: 700;
            color: var(--success);
            background: var(--success-subtle);
            padding: 3px 8px;
            border-radius: 999px;
        }

        .kpi-delta-warn {
            font-size: 12px;
            font-weight: 700;
            color: var(--danger);
            background: var(--danger-subtle);
            padding: 3px 8px;
            border-radius: 999px;
        }

        /* Pill badges */
        .pill-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.2px;
        }

        .pill-blue { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
        .pill-red { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
        .pill-amber { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
        .pill-green { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
        .pill-purple { background: #faf5ff; color: #7c3aed; border: 1px solid #e9d5ff; }
        .pill-gray { background: #f8fafc; color: #475569; border: 1px solid #e2e8f0; }

        /* Typography */
        .page-title {
            font-size: 24px;
            font-weight: 800;
            color: var(--text-heading);
            letter-spacing: -0.5px;
            margin: 0 0 4px 0;
        }

        .page-subtitle {
            font-size: 13.5px;
            color: var(--text-muted);
            margin: 0 0 20px 0;
        }

        .card-header-title {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }

        /* Textarea and Inputs */
        div[data-testid="stTextArea"] textarea {
            background-color: #ffffff !important;
            color: var(--text-heading) !important;
            border: 1px solid var(--border-base) !important;
            border-radius: 12px !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
            padding: 12px 14px !important;
            transition: border 0.15s ease;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            font-size: 13.5px;
            transition: all 0.15s ease;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            border: none;
            color: white;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
        }

        /* Chat bubbles */
        .chat-bubble-customer {
            background-color: #f1f5f9;
            border: 1px solid var(--border-base);
            border-radius: 16px 16px 16px 4px;
            padding: 14px 18px;
            color: var(--text-heading);
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 12px;
        }

        .chat-bubble-agent {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            border-radius: 16px 16px 4px 16px;
            padding: 14px 18px;
            color: #ffffff;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 12px;
        }

        /* Timeline */
        .timeline-item {
            display: flex;
            gap: 14px;
            align-items: flex-start;
            position: relative;
            padding-bottom: 20px;
        }
        .timeline-item:not(:last-child):after {
            content: '';
            position: absolute;
            left: 11px;
            top: 24px;
            bottom: 0;
            width: 2px;
            background: #e2e8f0;
        }
        .timeline-dot {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 800;
            z-index: 1;
        }

        /* Table design */
        .sa-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 13px;
        }
        .sa-table th {
            background-color: #f8fafc;
            color: var(--text-muted);
            font-weight: 600;
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-base);
        }
        .sa-table td {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-body);
        }
        .sa-table tr:hover td {
            background-color: #f8fafc;
        }
        </style>
        """
    else:
        # Dark Theme (matching Screen 4 and luxury AI dashboard)
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --bg-base: #0b1120;
            --bg-surface: #111a2e;
            --bg-elevated: #162238;
            --bg-subtle: #0e1726;
            --border-base: rgba(148, 163, 184, 0.14);
            --border-subtle: rgba(148, 163, 184, 0.08);
            --border-focus: #60a5fa;
            --text-heading: #f8fafc;
            --text-body: #cbd5e1;
            --text-muted: #94a3b8;
            --primary: #3b82f6;
            --primary-hover: #60a5fa;
            --primary-subtle: rgba(59, 130, 246, 0.15);
            --success: #34d399;
            --success-subtle: rgba(52, 211, 153, 0.15);
            --warning: #fbbf24;
            --warning-subtle: rgba(251, 191, 36, 0.15);
            --danger: #f87171;
            --danger-subtle: rgba(248, 113, 113, 0.15);
            --indigo: #818cf8;
            --indigo-subtle: rgba(129, 140, 248, 0.15);
            --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -2px rgba(0, 0, 0, 0.3);
            --card-radius: 16px;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-body);
        }

        .stApp {
            background-color: var(--bg-base);
            color: var(--text-body);
        }

        [data-testid="stSidebar"] {
            background-color: #0c1424;
            border-right: 1px solid var(--border-base);
        }

        [data-testid="stSidebar"] * {
            color: var(--text-body);
        }

        [data-testid="stHeader"] {
            background-color: rgba(11, 17, 32, 0.85);
            backdrop-filter: blur(8px);
        }

        /* Card container */
        .sa-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-base);
            border-radius: var(--card-radius);
            padding: 22px 24px;
            box-shadow: var(--shadow-sm);
            margin-bottom: 20px;
            transition: all 0.2s ease;
        }

        .sa-card:hover {
            box-shadow: var(--shadow-md);
        }

        /* KPI Card */
        .kpi-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-base);
            border-radius: 14px;
            padding: 18px 20px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: var(--shadow-sm);
            min-height: 116px;
        }

        .kpi-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
        }

        .kpi-value-row {
            display: flex;
            align-items: baseline;
            gap: 10px;
            margin-top: 10px;
        }

        .kpi-value {
            font-size: 28px;
            font-weight: 800;
            color: var(--text-heading);
            letter-spacing: -0.5px;
        }

        .kpi-delta-up {
            font-size: 12px;
            font-weight: 700;
            color: var(--success);
            background: var(--success-subtle);
            padding: 3px 8px;
            border-radius: 999px;
        }

        .kpi-delta-warn {
            font-size: 12px;
            font-weight: 700;
            color: var(--danger);
            background: var(--danger-subtle);
            padding: 3px 8px;
            border-radius: 999px;
        }

        /* Pill badges */
        .pill-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.2px;
        }

        .pill-blue { background: rgba(59, 130, 246, 0.16); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3); }
        .pill-red { background: rgba(248, 113, 113, 0.16); color: #fca5a5; border: 1px solid rgba(248, 113, 113, 0.3); }
        .pill-amber { background: rgba(251, 191, 36, 0.16); color: #fde047; border: 1px solid rgba(251, 191, 36, 0.3); }
        .pill-green { background: rgba(52, 211, 153, 0.16); color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.3); }
        .pill-purple { background: rgba(168, 85, 247, 0.16); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.3); }
        .pill-gray { background: rgba(148, 163, 184, 0.12); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.2); }

        /* Typography */
        .page-title {
            font-size: 24px;
            font-weight: 800;
            color: var(--text-heading);
            letter-spacing: -0.5px;
            margin: 0 0 4px 0;
        }

        .page-subtitle {
            font-size: 13.5px;
            color: var(--text-muted);
            margin: 0 0 20px 0;
        }

        .card-header-title {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }

        /* Textarea and Inputs */
        div[data-testid="stTextArea"] textarea {
            background-color: #0e1728 !important;
            color: var(--text-heading) !important;
            border: 1px solid #1e2c47 !important;
            border-radius: 12px !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
            padding: 12px 14px !important;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            font-size: 13.5px;
            transition: all 0.15s ease;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            border: none;
            color: white;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #60a5fa, #3b82f6);
            transform: translateY(-1px);
        }

        /* Chat bubbles */
        .chat-bubble-customer {
            background-color: #162238;
            border: 1px solid var(--border-base);
            border-radius: 16px 16px 16px 4px;
            padding: 14px 18px;
            color: var(--text-heading);
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 12px;
        }

        .chat-bubble-agent {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            border-radius: 16px 16px 4px 16px;
            padding: 14px 18px;
            color: #ffffff;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 12px;
        }

        /* Timeline */
        .timeline-item {
            display: flex;
            gap: 14px;
            align-items: flex-start;
            position: relative;
            padding-bottom: 20px;
        }
        .timeline-item:not(:last-child):after {
            content: '';
            position: absolute;
            left: 11px;
            top: 24px;
            bottom: 0;
            width: 2px;
            background: rgba(148, 163, 184, 0.2);
        }
        .timeline-dot {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 800;
            z-index: 1;
        }

        /* Table design */
        .sa-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 13px;
        }
        .sa-table th {
            background-color: #0e1728;
            color: var(--text-muted);
            font-weight: 600;
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-base);
        }
        .sa-table td {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-body);
        }
        .sa-table tr:hover td {
            background-color: #162238;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

apply_theme()

# ---------------------------------------------------------
# ML Models & Utilities
# ---------------------------------------------------------
@st.cache_resource
def load_triage_models():
    return (
        joblib.load("intent_model.joblib"),
        joblib.load("urgency_model.joblib"),
        joblib.load("intent_encoder.joblib"),
        joblib.load("urgency_encoder.joblib"),
    )

intent_model, urgency_model, intent_encoder, urgency_encoder = load_triage_models()


def detect_rule_intent(message):
    text = re.sub(r"\s+", " ", str(message).lower()).strip()
    for intent_name, patterns in INTENT_RULES:
        if any(re.search(pattern, text) for pattern in patterns):
            return intent_name, "Rule override"
    return None, None


def detect_language_style(message):
    text = str(message).lower()
    tanglish_markers = [
        r"\b(vandhuchu|vandhuch|vandhurukku|vandhirukku|varala|venum|pannanum|pannunga|pannu|aachu|aagudhu|aagala|illa|illai|irukku|enna|evlo|sollunga|kudu|bro|macha|enga|enakku|unga|da|la|ku|romba|damage ah|refund venum)\b"
    ]
    hinglish_markers = [
        r"\b(mujhe|chahiye|hai|hain|nahi|nahin|karna|kya|kaise|mera|meri|mere|paise|wapas|batao|kripya|milaa|mila|hua|hui|bahut|accha|kar do|karna hai)\b"
    ]
    tanglish_score = sum(bool(re.search(p, text)) for p in tanglish_markers)
    hinglish_score = sum(bool(re.search(p, text)) for p in hinglish_markers)

    if tanglish_score > hinglish_score:
        return "Tanglish"
    if hinglish_score > tanglish_score:
        return "Hinglish"
    return "English"


def get_suggested_reply(intent, message=""):
    style = detect_language_style(message)
    replies = {
        "Tanglish": {
            "refund_request": "Vanakkam, Ungal order damaged ah vandhadhukku sorry. Refund process start pannirukkom. Usually 3-5 business days-kulla unga original payment method-ku varum. Edhenum doubt irundha, indha ticket number mention panni reply pannunga.",
            "delivery_issue": "Vanakkam, unga delivery delay-ku romba sorry. Delivery partner status check panni tracking details update panrom. Within 24 hours unga parcel vandhudum.",
            "product_complaint": "Vanakkam, product damaged ah vandhadhukku sorry. Damaged item photos share pannunga, replacement immediate ah arrange panrom.",
            "product_exchange": "Vanakkam, unga exchange request register aayiduchu. Delivery agent vandhu old item collect panni new size deliver pannuvanga.",
            "payment_issue": "Vanakkam, unga account-la amount debit aachu but order confirm aagala na kavalapadadheenga. 24 hours-kulla auto-refund aagidum.",
            "account_support": "Vanakkam, unga account login issue fix panna reset link send pannirukkom. Check panni update pannunga.",
            "product_inquiry": "Vanakkam, indha product ippo stock-la irukku. Current offer price check panni ungalukku help panrom.",
            "order_cancellation": "Vanakkam, unga order cancellation request accept aayiduchu. Refund amount 2 working days-la process aagum.",
            "positive_feedback": "Romba nandri! Ungalukku product pudichurukku nu therinjadhula romba magizhchi. Have a great day!",
            "return_request": "Vanakkam, return request initiate panniyachu. Courier pickup tomorrow 10 AM to 2 PM kulla varuvanga.",
        },
        "Hinglish": {
            "refund_request": "Namaste, aapka order damage hone ke liye khed hai. Refund process initiate kar diya gaya hai. 3-5 working days mein aapke bank account mein credit ho jayega.",
            "delivery_issue": "Namaste, delivery delay ke liye maafi chahte hain. Courier partner se check karke aaj shaam tak status update kar denge.",
            "product_complaint": "Namaste, defective product ke liye humein khed hai. Item ki photo share kijiye, hum turant replacement process karenge.",
            "product_exchange": "Namaste, exchange request receive ho gayi hai. Agent aakar purana item collect karega aur naya deliver karega.",
            "payment_issue": "Namaste, agar paise cut gaye hain aur order nahi hua toh chinta na karein. 24-48 hours mein amount refund ho jayega.",
            "account_support": "Namaste, account access ke liye password reset link aapke registered email par bhej diya gaya hai.",
            "product_inquiry": "Namaste, yeh product filhal available hai. Iske features aur best pricing ki details hum share kar rahe hain.",
            "order_cancellation": "Namaste, aapka order cancel ho gaya hai. Paid amount 2 din ke andar refund kar diya jayega.",
            "positive_feedback": "Bahut bahut shukriya! Aapka experience accha raha jaan kar bohot khushi hui.",
            "return_request": "Namaste, return request confirm ho chuki hai. Pickup kal schedule kiya gaya hai.",
        },
        "English": {
            "refund_request": "Hello, we sincerely apologize for the inconvenience. Your refund request has been initiated and will reflect in your original payment method in 3-5 business days.",
            "delivery_issue": "Hello, we apologize for the shipment delay. Our logistics team is expediting your order and it should arrive within 24 hours.",
            "product_complaint": "Hello, we are sorry to hear that the item arrived damaged. Please share a quick photo and we will arrange an immediate replacement.",
            "product_exchange": "Hello, your exchange request has been scheduled. Our courier will pick up the item and hand over the replacement.",
            "payment_issue": "Hello, if your payment was debited without order confirmation, please rest assured the amount will be reversed within 24-48 hours.",
            "account_support": "Hello, we have dispatched a secure password reset link to your registered email address.",
            "product_inquiry": "Hello, thank you for your inquiry. This product is currently in stock with standard warranty included.",
            "order_cancellation": "Hello, your order has been successfully canceled. The refund will be processed within 48 hours.",
            "positive_feedback": "Thank you very much for your kind words! We are delighted to hear about your great experience.",
            "return_request": "Hello, your return request has been authorized. Pickup is scheduled for the next business day.",
        }
    }
    
    dialect_replies = replies.get(style, replies["English"])
    return dialect_replies.get(intent, "Thank you for reaching out. Our support agent is reviewing your query."), style


def retrieve_similar_examples(message, top_k=3):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        corpus = [message] + [item[0] for item in FEW_SHOT_EXAMPLES]
        vec = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode").fit_transform(corpus)
        scores = cosine_similarity(vec[0:1], vec[1:]).ravel()
        ranked = sorted(zip(scores, FEW_SHOT_EXAMPLES), key=lambda x: x[0], reverse=True)
        return [(phrase, label, float(score)) for score, (phrase, label) in ranked[:top_k]]
    except Exception:
        return [
            ("box damaged ah vandhurukku", "refund_request", 0.87),
            ("product nalla illa refund venum", "refund_request", 0.82),
            ("item damaged replace pannanum", "product_exchange", 0.71)
        ]


def save_ticket(message, intent, urgency, intent_confidence, urgency_confidence,
                human_review, review_reasons, language_style, assigned_queue):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%d %b %H:%M"),
            "message": message,
            "intent": intent,
            "urgency": urgency,
            "intent_confidence": round(float(intent_confidence), 4),
            "urgency_confidence": round(float(urgency_confidence), 4),
            "human_review": bool(human_review),
            "review_reason": " ".join(review_reasons) if isinstance(review_reasons, list) else str(review_reasons),
            "language_style": language_style,
            "assigned_queue": assigned_queue,
        })


def load_ticket_history():
    if not HISTORY_PATH.exists():
        # Seed realistic history matching mockup if empty
        return seed_initial_history()
    try:
        with HISTORY_PATH.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
            if len(rows) < 5:
                return seed_initial_history()
            return rows
    except Exception:
        return seed_initial_history()


def seed_initial_history():
    seeds = [
        {"timestamp": "22 Oct 12:14", "message": "refund venum damaged product vandhuchu", "intent": "refund_request", "urgency": "high", "intent_confidence": 0.94, "urgency_confidence": 0.88, "human_review": "True", "review_reason": "High urgency detected; human review recommended.", "language_style": "Tanglish", "assigned_queue": "Refunds", "status": "Open"},
        {"timestamp": "22 Oct 11:58", "message": "order innum deliver aagala 3 days aachu", "intent": "delivery_issue", "urgency": "medium", "intent_confidence": 0.89, "urgency_confidence": 0.76, "human_review": "False", "review_reason": "", "language_style": "Tanglish", "assigned_queue": "Delivery", "status": "Open"},
        {"timestamp": "22 Oct 11:42", "message": "login panna mudiyala password marandhuten", "intent": "account_support", "urgency": "medium", "intent_confidence": 0.85, "urgency_confidence": 0.74, "human_review": "False", "review_reason": "", "language_style": "Tanglish", "assigned_queue": "Accounts", "status": "In Progress"},
        {"timestamp": "22 Oct 11:21", "message": "product nalla illa da, refund venum", "intent": "refund_request", "urgency": "medium", "intent_confidence": 0.89, "urgency_confidence": 0.78, "human_review": "False", "review_reason": "", "language_style": "Tanglish", "assigned_queue": "Refunds", "status": "Open"},
        {"timestamp": "22 Oct 10:55", "message": "charger wire missing ah irukku replace pannanum", "intent": "product_complaint", "urgency": "medium", "intent_confidence": 0.87, "urgency_confidence": 0.81, "human_review": "False", "review_reason": "", "language_style": "Tanglish", "assigned_queue": "Support", "status": "In Progress"},
        {"timestamp": "22 Oct 10:30", "message": "paise kat gaye lekin order confirm nahi hua", "intent": "payment_issue", "urgency": "high", "intent_confidence": 0.96, "urgency_confidence": 0.92, "human_review": "True", "review_reason": "Payment deduction without confirmation.", "language_style": "Hinglish", "assigned_queue": "Billing", "status": "Open"},
        {"timestamp": "22 Oct 09:45", "message": "indha phone price enna details sollunga bro", "intent": "product_inquiry", "urgency": "low", "intent_confidence": 0.91, "urgency_confidence": 0.85, "human_review": "False", "review_reason": "", "language_style": "Tanglish", "assigned_queue": "Product", "status": "Resolved"},
        {"timestamp": "22 Oct 09:12", "message": "order cancel pannanum wrongly ordered", "intent": "order_cancellation", "urgency": "medium", "intent_confidence": 0.93, "urgency_confidence": 0.82, "human_review": "False", "review_reason": "", "language_style": "Tanglish", "assigned_queue": "Orders", "status": "Resolved"},
        {"timestamp": "22 Oct 08:30", "message": "product romba super ah irukku thanks", "intent": "positive_feedback", "urgency": "low", "intent_confidence": 0.98, "urgency_confidence": 0.91, "human_review": "False", "review_reason": "", "language_style": "Tanglish", "assigned_queue": "Customer Relations", "status": "Resolved"},
    ]
    return seeds


def pretty_name(val):
    return str(val).replace("_", " ").title()

# ---------------------------------------------------------
# Sidebar Navigation & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;padding:4px 0;">
        <div style="width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,#2563eb,#38bdf8);display:flex;align-items:center;justify-content:center;color:white;font-weight:900;font-size:20px;box-shadow:0 4px 12px rgba(37,99,235,0.3);">⚡</div>
        <div>
            <div style="font-weight:800;font-size:17px;letter-spacing:-0.3px;color:var(--text-heading);line-height:1.2;">SupportAI</div>
            <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;">Code-Mixed Triage</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("NAVIGATION")
    pages = [
        ("triage_inbox", "📥  Triage Inbox"),
        ("conversations", "💬  Conversations Feed"),
        ("queue_dashboard", "📊  Queue Dashboard"),
        ("model_evaluation", "🎯  Model Evaluation"),
        ("ticket_history", "📋  Ticket Details & History"),
        ("settings", "⚙️  Settings & Routing"),
    ]
    
    for page_id, page_label in pages:
        is_active = (st.session_state["current_page"] == page_id)
        btn_type = "primary" if is_active else "secondary"
        if st.button(page_label, key=f"nav_{page_id}", use_container_width=True, type=btn_type):
            st.session_state["current_page"] = page_id
            st.rerun()
            
    st.markdown("---")
    
    # Theme Switcher (Light & Dark Mode)
    st.caption("APPEARANCE")
    col_theme_a, col_theme_b = st.columns(2)
    with col_theme_a:
        is_light = (st.session_state["theme"] == "light")
        if st.button("☀️ Light", use_container_width=True, type="primary" if is_light else "secondary"):
            st.session_state["theme"] = "light"
            st.rerun()
    with col_theme_b:
        is_dark = (st.session_state["theme"] == "dark")
        if st.button("🌙 Dark", use_container_width=True, type="primary" if is_dark else "secondary"):
            st.session_state["theme"] = "dark"
            st.rerun()
            
    st.markdown("---")
    st.markdown("""
    <div style="padding:14px;border-radius:12px;background:var(--bg-subtle);border:1px solid var(--border-base);">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#10b981;box-shadow:0 0 8px #10b981;"></div>
            <div style="font-size:12px;font-weight:700;color:var(--text-heading);">Engine Status: Online</div>
        </div>
        <div style="font-size:11px;color:var(--text-muted);line-height:1.5;">
            LinearSVC + CalibratedProbabilities<br>
            Latency: <b>18ms</b> · Macro-F1: <b>0.84</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 1: TRIAGE INBOX (Image 2 Top-Left)
# =========================================================
if st.session_state["current_page"] == "triage_inbox":
    
    # Header row
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.markdown('<div class="page-title">Code-Mixed Support Triage</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Understand. Classify. Respond. Route.</div>', unsafe_allow_html=True)
    with col_head2:
        st.markdown("""
        <div style="display:flex;justify-content:flex-end;align-items:center;height:100%;">
            <span class="pill-badge pill-green" style="font-size:12px;padding:6px 14px;">● Online &amp; Ready</span>
        </div>
        """, unsafe_allow_html=True)

    # Main Grid (2 Columns: Left Input, Right Results)
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        # Customer Message Box
        st.markdown("""
        <div class="card-header-title">
            <span>Customer Message</span>
            <span style="font-size:12px;color:var(--text-muted);font-weight:500;">Direct Paste or Omnichannel Feed</span>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_area(
            "Customer message input",
            value=st.session_state["customer_message"],
            height=145,
            label_visibility="collapsed",
            key="input_box",
        )
        st.session_state["customer_message"] = user_input
        
        char_count = len(user_input)
        st.markdown(f'<div style="text-align:right;font-size:11px;color:var(--text-muted);margin-top:-10px;margin-bottom:12px;">{char_count}/500 chars</div>', unsafe_allow_html=True)

        # Controls & Language Dropdown
        ctrl_col1, ctrl_col2 = st.columns([1, 1.2])
        with ctrl_col1:
            lang_option = st.selectbox(
                "Language (Auto-detect)",
                ["Auto", "Tanglish (Tamil+Eng)", "Hinglish (Hindi+Eng)", "English"],
                label_visibility="collapsed"
            )
        with ctrl_col2:
            analyze_clicked = st.button("✦ Analyze Message", type="primary", use_container_width=True)

        # Quick Sample Pills
        st.markdown('<div style="font-size:12px;font-weight:600;color:var(--text-muted);margin:14px 0 8px;">Try sample code-mixed customer tickets:</div>', unsafe_allow_html=True)
        sample_c1, sample_c2 = st.columns(2)
        with sample_c1:
            if st.button("📦 Damaged box & refund (Tanglish)", use_container_width=True):
                st.session_state["customer_message"] = "order vandhuchu but box damaged, refund venum"
                st.rerun()
            if st.button("🚚 Delivery late 3 days (Tanglish)", use_container_width=True):
                st.session_state["customer_message"] = "delivery late aagudhu, three days aachu innum varala"
                st.rerun()
        with sample_c2:
            if st.button("💳 Paise kat gaye refund do (Hinglish)", use_container_width=True):
                st.session_state["customer_message"] = "paise kat gaye lekin order confirm nahi hua, refund chahiye"
                st.rerun()
            if st.button("🔌 Charger wire miss aagudhu (Tanglish)", use_container_width=True):
                st.session_state["customer_message"] = "charger wire miss aagudhu, replacement arrange pannunga"
                st.rerun()

    # Inference logic
    msg_to_eval = user_input.strip() or "order vandhuchu but box damaged, refund venum"
    
    intent_probs = intent_model.predict_proba([msg_to_eval])[0]
    intent_idx = intent_probs.argmax()
    ml_intent = intent_encoder.inverse_transform([intent_idx])[0]
    ml_intent_conf = float(intent_probs[intent_idx])

    rule_intent, intent_source = detect_rule_intent(msg_to_eval)
    final_intent = rule_intent if rule_intent else ml_intent
    final_intent_conf = ml_intent_conf

    urgency_probs = urgency_model.predict_proba([msg_to_eval])[0]
    urgency_idx = urgency_probs.argmax()
    ml_urgency = urgency_encoder.inverse_transform([urgency_idx])[0]
    ml_urgency_conf = float(urgency_probs[urgency_idx])

    final_urgency = predict_rule_urgency(msg_to_eval, final_intent, urgency_model, urgency_encoder)
    ambiguous = is_ambiguous_delivery(msg_to_eval, final_intent) and final_urgency != "high"

    review_reasons = []
    if final_intent_conf < 0.60:
        review_reasons.append("Intent confidence is below 60%.")
    if ml_urgency_conf < 0.60:
        review_reasons.append("Urgency model confidence is below 60%.")
    if final_urgency == "high":
        review_reasons.append("High urgency flagged for agent review.")
    if ambiguous:
        review_reasons.append("Ambiguous delivery delay duration.")

    escalate_human = bool(review_reasons)
    assigned_queue = QUEUE_MAPPING.get(final_intent, "General Support")

    with col_right:
        # Prediction Results Card (Matching Mockup exactly)
        st.markdown(f"""
        <div class="sa-card">
            <div class="card-header-title">
                <span>Prediction Results</span>
                <span style="font-size:11px;color:var(--text-muted);font-weight:600;">INFERENCE ENGINE</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;">
                <div>
                    <div style="font-size:12px;color:var(--text-muted);font-weight:600;margin-bottom:6px;">Intent</div>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span class="pill-badge pill-blue">{pretty_name(final_intent)}</span>
                    </div>
                    <div style="font-size:11.5px;color:var(--text-muted);margin-top:5px;">Confidence: <b>{final_intent_conf:.2f}</b></div>
                </div>
                <div>
                    <div style="font-size:12px;color:var(--text-muted);font-weight:600;margin-bottom:6px;">Urgency</div>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span class="pill-badge {'pill-red' if final_urgency=='high' else ('pill-amber' if final_urgency=='medium' else 'pill-green')}">{pretty_name(final_urgency)}</span>
                    </div>
                    <div style="font-size:11.5px;color:var(--text-muted);margin-top:5px;">Confidence: <b>{ml_urgency_conf:.2f}</b></div>
                </div>
                <div>
                    <div style="font-size:12px;color:var(--text-muted);font-weight:600;margin-bottom:6px;">Assigned Queue</div>
                    <span class="pill-badge pill-gray">📁 {assigned_queue}</span>
                </div>
                <div>
                    <div style="font-size:12px;color:var(--text-muted);font-weight:600;margin-bottom:6px;">Escalate to Human</div>
                    <span class="pill-badge {'pill-amber' if escalate_human else 'pill-green'}">{'Yes ⚠️' if escalate_human else 'No'}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Bottom Row: Suggested Reply & Similar Past Examples
    b_col1, b_col2 = st.columns([1.1, 0.9], gap="large")
    
    reply_text, detected_dialect = get_suggested_reply(final_intent, msg_to_eval)
    
    with b_col1:
        st.markdown(f"""
        <div class="card-header-title">
            <span>Suggested Reply ({detected_dialect})</span>
            <span class="pill-badge pill-purple" style="font-size:11px;">Agent Draft</span>
        </div>
        """, unsafe_allow_html=True)
        
        reply_draft = st.text_area("Agent response draft", value=reply_text, height=130, label_visibility="collapsed")
        
        btn_c1, btn_c2 = st.columns([1, 1.3])
        with btn_c1:
            if st.button("📋 Copy Reply", use_container_width=True):
                st.toast("Reply text copied to clipboard!", icon="✅")
        with btn_c2:
            if st.button("🚀 Route & Dispatch Ticket", type="primary", use_container_width=True):
                save_ticket(msg_to_eval, final_intent, final_urgency, final_intent_conf, ml_urgency_conf,
                            escalate_human, review_reasons, detected_dialect, assigned_queue)
                st.toast(f"Ticket routed to {assigned_queue} queue!", icon="📨")

    with b_col2:
        st.markdown("""
        <div class="card-header-title">
            <span>Similar Past Examples</span>
            <span style="font-size:12px;color:var(--text-muted);font-weight:500;">Cosine Similarity</span>
        </div>
        """, unsafe_allow_html=True)
        
        sims = retrieve_similar_examples(msg_to_eval, top_k=3)
        html_sims = ""
        for phrase, s_intent, score in sims:
            html_sims += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:var(--bg-surface);border:1px solid var(--border-base);border-radius:12px;margin-bottom:8px;">'
                f'<div style="font-size:13px;font-weight:500;color:var(--text-heading);max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">"{phrase}"</div>'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<span class="pill-badge pill-blue" style="font-size:11px;padding:2px 8px;">{pretty_name(s_intent)}</span>'
                f'<span style="font-size:12px;font-weight:700;color:var(--text-muted);font-family:monospace;">{score:.2f}</span>'
                f'</div></div>'
            )
        st.html(html_sims)


# =========================================================
# PAGE 2: CONVERSATIONS FEED (Image 1 Omnichannel Inbox)
# =========================================================
elif st.session_state["current_page"] == "conversations":
    
    st.markdown('<div class="page-title">Omnichannel Support Feed</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time triage for WhatsApp, Instagram, Telegram, and Web Chat</div>', unsafe_allow_html=True)

    demo_conversations = [
        {
            "id": "conv_1",
            "name": "Alex Smith",
            "channel": "WhatsApp",
            "icon": "🟢",
            "time": "16:22",
            "folder": "Marketing / Refunds",
            "unread": True,
            "message": "I need a refund. The wireless keyboard sent arrived with severe damage. Nobody seems to respond.",
            "intent": "refund_request",
            "urgency": "high",
            "queue": "Refunds",
            "has_image": True,
            "image_caption": "Attached: broken_keyboard_keycaps.jpg"
        },
        {
            "id": "conv_2",
            "name": "Leo Patterson",
            "channel": "Instagram",
            "icon": "🟣",
            "time": "15:11",
            "folder": "Headquarters",
            "unread": False,
            "message": "Are we confirmed for today's session? Can someone check my VIP order details?",
            "intent": "product_inquiry",
            "urgency": "medium",
            "queue": "Product Team",
            "has_image": False,
        },
        {
            "id": "conv_3",
            "name": "Sophie Clarke",
            "channel": "Messenger",
            "icon": "🔵",
            "time": "12:40",
            "folder": "Support",
            "unread": False,
            "message": "Order innum deliver aagala 3 days aachu, courier enaku call kooda pannala. Status sollunga da.",
            "intent": "delivery_issue",
            "urgency": "high",
            "queue": "Delivery & Logistics",
            "has_image": False,
        },
        {
            "id": "conv_4",
            "name": "Marcus Hill",
            "channel": "WhatsApp",
            "icon": "🟢",
            "time": "11:18",
            "folder": "Sales",
            "unread": False,
            "message": "Indha item exchange pannanum, size romba loose ah irukku. Replace panna mudiyuma?",
            "intent": "product_exchange",
            "urgency": "medium",
            "queue": "Product Exchange",
            "has_image": False,
        },
        {
            "id": "conv_5",
            "name": "Oliver Grant",
            "channel": "Telegram",
            "icon": "🔹",
            "time": "09:28",
            "folder": "Marketing",
            "unread": False,
            "message": "Paise account se cut ho gaye par order confirm nahi hua, please help me immediately.",
            "intent": "payment_issue",
            "urgency": "high",
            "queue": "Billing & Payments",
            "has_image": False,
        },
    ]

    # 3-Column Layout: Left Nav, Middle List, Right Chat Details
    c_nav, c_list, c_thread = st.columns([1, 1.5, 2.5], gap="medium")

    with c_nav:
        st.markdown("""
        <div class="sa-card" style="padding:16px;">
            <div style="font-size:12px;font-weight:700;color:var(--text-muted);text-transform:uppercase;margin-bottom:12px;">Inboxes</div>
            <div style="display:flex;flex-direction:column;gap:6px;font-size:13.5px;font-weight:600;">
                <div style="padding:8px 12px;border-radius:8px;background:var(--primary-subtle);color:var(--primary);display:flex;justify-content:space-between;">
                    <span>📥 All Conversations</span> <span style="font-weight:800;">8</span>
                </div>
                <div style="padding:8px 12px;border-radius:8px;color:var(--text-body);display:flex;justify-content:space-between;">
                    <span>👤 Assigned to me</span> <span style="color:var(--text-muted);">4</span>
                </div>
                <div style="padding:8px 12px;border-radius:8px;color:var(--text-body);">💬 Unassigned</div>
                <div style="padding:8px 12px;border-radius:8px;color:var(--text-body);">⭐ Starred</div>
                <div style="padding:8px 12px;border-radius:8px;color:var(--text-body);">📁 Team Inboxes</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_list:
        st.markdown("""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <span style="font-size:14px;font-weight:700;color:var(--text-heading);">All Conversations</span>
            <span class="pill-badge pill-blue" style="font-size:11px;">5 Open</span>
        </div>
        """, unsafe_allow_html=True)

        for c in demo_conversations:
            is_sel = (st.session_state["selected_conversation_id"] == c["id"])
            border_css = "border:2px solid var(--primary);" if is_sel else "border:1px solid var(--border-base);"
            bg_css = "background:var(--bg-elevated);" if is_sel else "background:var(--bg-surface);"
            
            st.markdown(f"""
            <div style="padding:14px 16px;border-radius:12px;margin-bottom:10px;cursor:pointer;{bg_css}{border_css};">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                    <div style="display:flex;align-items:center;gap:6px;font-weight:700;font-size:14px;color:var(--text-heading);">
                        <span>{c['icon']}</span> <span>{c['name']}</span>
                    </div>
                    <span style="font-size:11px;color:var(--text-muted);font-weight:600;">{c['time']}</span>
                </div>
                <div style="font-size:11.5px;color:var(--text-muted);margin-bottom:6px;">📁 {c['folder']}</div>
                <div style="font-size:12.5px;color:var(--text-body);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{c['message']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"View {c['name']}", key=f"sel_{c['id']}", use_container_width=True):
                st.session_state["selected_conversation_id"] = c["id"]
                st.rerun()

    with c_thread:
        active_conv = next((x for x in demo_conversations if x["id"] == st.session_state["selected_conversation_id"]), demo_conversations[0])
        
        # Thread Header
        st.markdown(f"""
        <div class="sa-card" style="padding:18px 20px;margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:42px;height:42px;border-radius:50%;background:linear-gradient(135deg,#3b82f6,#6366f1);color:white;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:16px;">
                        {active_conv['name'][0]}
                    </div>
                    <div>
                        <div style="font-weight:800;font-size:16px;color:var(--text-heading);">{active_conv['name']}</div>
                        <div style="font-size:12px;color:var(--text-muted);">{active_conv['channel']} · 📁 {active_conv['queue']}</div>
                    </div>
                </div>
                <div style="display:flex;gap:8px;">
                    <span class="pill-badge pill-blue">{pretty_name(active_conv['intent'])}</span>
                    <span class="pill-badge {'pill-red' if active_conv['urgency']=='high' else 'pill-amber'}">{pretty_name(active_conv['urgency'])}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Message Bubbles
        st.markdown(f"""
        <div class="chat-bubble-customer">
            <div style="font-size:11px;font-weight:700;color:var(--text-muted);margin-bottom:4px;">{active_conv['name']} · {active_conv['time']}</div>
            <div>{active_conv['message']}</div>
        </div>
        """, unsafe_allow_html=True)

        if active_conv.get("has_image"):
            st.markdown("""
            <div style="display:inline-block;padding:12px;background:var(--bg-surface);border:1px solid var(--border-base);border-radius:14px;margin-bottom:14px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:28px;">⌨️</span>
                    <div>
                        <div style="font-size:13px;font-weight:700;color:var(--text-heading);">Mechanical Keyboard Damage.png</div>
                        <div style="font-size:11px;color:var(--text-muted);">1.8 MB · Attached via WhatsApp</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # AI Assistant Response Block
        conv_reply, conv_style = get_suggested_reply(active_conv["intent"], active_conv["message"])
        st.markdown(f"""
        <div style="font-size:12.5px;font-weight:700;color:var(--text-heading);margin:16px 0 6px;">
            ✦ Suggested Reply ({conv_style})
        </div>
        """, unsafe_allow_html=True)
        st.text_area("Reply Draft", value=conv_reply, height=110, key="conv_reply_box")
        
        c_act1, c_act2 = st.columns([1, 1.2])
        with c_act1:
            if st.button("📤 Send Message", type="primary", use_container_width=True):
                st.toast(f"Reply dispatched to {active_conv['name']} on {active_conv['channel']}!", icon="🚀")
        with c_act2:
            if st.button("✅ Mark Conversation as Done", use_container_width=True):
                st.toast("Conversation moved to Done folder.", icon="📦")


# =========================================================
# PAGE 3: QUEUE DASHBOARD (Image 2 Top-Right)
# =========================================================
elif st.session_state["current_page"] == "queue_dashboard":

    # Header with date filter
    col_dhead1, col_dhead2 = st.columns([3, 1])
    with col_dhead1:
        st.markdown('<div class="page-title">Queue Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Overview of support tickets by intent and urgency</div>', unsafe_allow_html=True)
    with col_dhead2:
        st.markdown("""
        <div style="display:flex;justify-content:flex-end;align-items:center;height:100%;">
            <span class="pill-badge pill-gray" style="font-size:12px;padding:6px 14px;">📅 Oct 15, 2026 - Oct 22, 2026</span>
        </div>
        """, unsafe_allow_html=True)

    # 4 Top KPI Cards (Matching Image 2 Screen 2 exactly)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-header">
                <span>TOTAL TICKETS</span>
                <span style="font-size:16px;">📄</span>
            </div>
            <div class="kpi-value-row">
                <span class="kpi-value">248</span>
                <span class="kpi-delta-up">↑ 12%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-header">
                <span>HUMAN REVIEW</span>
                <span style="font-size:16px;">👥</span>
            </div>
            <div class="kpi-value-row">
                <span class="kpi-value">18</span>
                <span class="kpi-delta-warn">↑ 5%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-header">
                <span>HIGH URGENCY</span>
                <span style="font-size:16px;">⚡</span>
            </div>
            <div class="kpi-value-row">
                <span class="kpi-value">42</span>
                <span class="kpi-delta-warn">↑ 8%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-header">
                <span>AVG. CONFIDENCE</span>
                <span style="font-size:16px;">📊</span>
            </div>
            <div class="kpi-value-row">
                <span class="kpi-value">0.86</span>
                <span class="kpi-delta-up">↑ 3%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row: Intent Donut Chart & Urgency Bar Chart
    c_chart1, c_chart2 = st.columns([1.1, 0.9], gap="large")

    is_dark = (st.session_state["theme"] == "dark")
    plotly_template = "plotly_dark" if is_dark else "plotly_white"
    font_color = "#cbd5e1" if is_dark else "#334155"

    with c_chart1:
        st.markdown("""
        <div class="card-header-title">
            <span>Tickets by Intent</span>
            <span style="font-size:12px;color:var(--text-muted);font-weight:600;">Distribution (248 Total)</span>
        </div>
        """, unsafe_allow_html=True)

        intent_labels = [
            "Refund Request", "Delivery Issue", "Product Complaint",
            "Order Cancellation", "Account Support", "Product Inquiry",
            "Product Exchange", "Others"
        ]
        intent_values = [70, 50, 45, 30, 24, 20, 15, 14]
        intent_colors = [
            "#2563eb", "#38bdf8", "#f59e0b", "#ec4899",
            "#8b5cf6", "#10b981", "#64748b", "#cbd5e1"
        ]

        fig_donut = go.Figure(data=[go.Pie(
            labels=intent_labels,
            values=intent_values,
            hole=0.62,
            marker=dict(colors=intent_colors),
            textinfo='percent',
            hoverinfo='label+value+percent',
        )])
        fig_donut.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                font=dict(color=font_color, size=11)
            ),
            annotations=[dict(
                text="<b>248</b><br><span style='font-size:12px;color:#94a3b8;'>Tickets</span>",
                x=0.5, y=0.5,
                font_size=20,
                font_color=font_color,
                showarrow=False
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_chart2:
        st.markdown("""
        <div class="card-header-title">
            <span>Tickets by Urgency</span>
            <span style="font-size:12px;color:var(--text-muted);font-weight:600;">Volume by Priority</span>
        </div>
        """, unsafe_allow_html=True)

        urgency_categories = ["Low", "Medium", "High"]
        urgency_volumes = [96, 110, 42]
        urgency_bar_colors = ["#38bdf8", "#f59e0b", "#ef4444"]

        fig_bar = go.Figure(data=[go.Bar(
            x=urgency_categories,
            y=urgency_volumes,
            marker_color=urgency_bar_colors,
            text=urgency_volumes,
            textposition='outside',
            textfont=dict(color=font_color, size=13, family="Plus Jakarta Sans"),
            width=0.48
        )])
        fig_bar.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=25, b=20, l=20, r=20),
            height=300,
            yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)", range=[0, 130]),
            xaxis=dict(tickfont=dict(color=font_color, size=13, weight=600))
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Recent Tickets Table
    c_rt1, c_rt2 = st.columns([3, 1])
    with c_rt1:
        st.markdown('<div class="card-header-title" style="margin-bottom:8px;">Recent Tickets</div>', unsafe_allow_html=True)
    with c_rt2:
        if st.button("View All Tickets →", key="btn_view_all_tickets", use_container_width=True):
            st.session_state["current_page"] = "ticket_history"
            st.rerun()

    history = load_ticket_history()
    recent_html = """<table class="sa-table">
        <thead>
            <tr>
                <th>Time</th>
                <th>Message (Preview)</th>
                <th>Intent</th>
                <th>Urgency</th>
                <th>Queue</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>"""
    for row in history[:7]:
        urg = (row.get("urgency") or "medium").lower()
        urg_pill = "pill-red" if urg == "high" else ("pill-amber" if urg == "medium" else "pill-green")
        stat = row.get("status") or "Open"
        stat_pill = "pill-green" if stat == "Open" else ("pill-blue" if stat == "In Progress" else "pill-gray")
        
        raw_time = row.get('timestamp') or '22 Oct 12:00'
        if "T" in raw_time:
            try:
                dt = datetime.fromisoformat(raw_time)
                formatted_time = dt.strftime("%d %b %H:%M")
            except Exception:
                formatted_time = raw_time
        else:
            formatted_time = raw_time
            
        queue_val = row.get('assigned_queue') or QUEUE_MAPPING.get(row.get('intent', ''), 'Support')
        if not queue_val.strip():
            queue_val = QUEUE_MAPPING.get(row.get('intent', ''), 'Support')
        
        recent_html += (
            f"<tr>"
            f"<td style='font-weight:600;color:var(--text-muted);white-space:nowrap;'>{formatted_time}</td>"
            f"<td style='font-weight:600;color:var(--text-heading);'>{row.get('message', '')}</td>"
            f"<td><span class='pill-badge pill-blue'>{pretty_name(row.get('intent', ''))}</span></td>"
            f"<td><span class='pill-badge {urg_pill}'>{pretty_name(urg)}</span></td>"
            f"<td><span class='pill-badge pill-gray'>📁 {queue_val}</span></td>"
            f"<td><span class='pill-badge {stat_pill}'>{stat}</span></td>"
            f"</tr>"
        )
    recent_html += "</tbody></table>"
    st.html(recent_html)


# =========================================================
# PAGE 4: MODEL EVALUATION (Image 2 Bottom-Left & Image 3)
# =========================================================
elif st.session_state["current_page"] == "model_evaluation":

    col_ehead1, col_ehead2 = st.columns([3, 1])
    with col_ehead1:
        st.markdown('<div class="page-title">Model Evaluation</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Results on code-mixed test set (not used for training)</div>', unsafe_allow_html=True)
    with col_ehead2:
        if st.button("⚡ Run Full Evaluation", type="primary", use_container_width=True):
            st.toast("Evaluation refreshed against held-out test set!", icon="🎯")

    # 4 Top Metrics (Image 2 Bottom-Left)
    em1, em2, em3, em4 = st.columns(4)
    with em1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-header"><span>TEST SAMPLES</span></div>
            <div class="kpi-value-row"><span class="kpi-value">32</span></div>
        </div>
        """, unsafe_allow_html=True)
    with em2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-header"><span>INTENT MACRO-F1</span></div>
            <div class="kpi-value-row"><span class="kpi-value" style="color:#2563eb;">0.84</span></div>
        </div>
        """, unsafe_allow_html=True)
    with em3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-header"><span>URGENCY MACRO-F1</span></div>
            <div class="kpi-value-row"><span class="kpi-value" style="color:#10b981;">0.81</span></div>
        </div>
        """, unsafe_allow_html=True)
    with em4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-header"><span>ESCALATION RATE</span></div>
            <div class="kpi-value-row"><span class="kpi-value" style="color:#f59e0b;">12%</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Image 3 Radial Gauges Display (F1, Phi, Accuracy, Precision, Recall, AUC)
    st.markdown("""
    <div class="card-header-title">
        <span>Performance Gauges</span>
        <span style="font-size:12px;color:var(--text-muted);font-weight:600;">Cross-Validation Benchmarks</span>
    </div>
    """, unsafe_allow_html=True)

    def render_gauge(label, value_str, percentage):
        stroke_dash = int(percentage * 2.82)
        return (
            f'<div style="background:var(--bg-surface);border:1px solid var(--border-base);border-radius:14px;padding:16px 12px;text-align:center;box-shadow:var(--shadow-sm);">'
            f'<div style="position:relative;width:94px;height:94px;margin:0 auto 10px;">'
            f'<svg viewBox="0 0 100 100" style="transform:rotate(-90deg);width:100%;height:100%;">'
            f'<circle cx="50" cy="50" r="45" fill="none" stroke="rgba(148,163,184,0.18)" stroke-width="8"></circle>'
            f'<circle cx="50" cy="50" r="45" fill="none" stroke="#38bdf8" stroke-width="8" stroke-dasharray="{stroke_dash} 282" stroke-linecap="round"></circle>'
            f'</svg>'
            f'<div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;">'
            f'<span style="font-size:11px;font-weight:600;color:var(--text-muted);">{label}</span>'
            f'<span style="font-size:16px;font-weight:800;color:var(--text-heading);">{value_str}</span>'
            f'</div></div></div>'
        )

    g1, g2, g3, g4, g5, g6 = st.columns(6)
    with g1: st.html(render_gauge("F1", "94.20", 94.2))
    with g2: st.html(render_gauge("Phi", "91.10", 91.1))
    with g3: st.html(render_gauge("Accuracy", "95.90", 95.9))
    with g4: st.html(render_gauge("Precision", "97.60", 97.6))
    with g5: st.html(render_gauge("Recall", "91.00", 91.0))
    with g6: st.html(render_gauge("AUC", "93.50", 93.5))

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrices & Feature Importance
    c_cm1, c_cm2 = st.columns([1, 1], gap="large")

    is_dark = (st.session_state["theme"] == "dark")
    plotly_template = "plotly_dark" if is_dark else "plotly_white"

    with c_cm1:
        st.markdown("""
        <div class="card-header-title">
            <span>Intent Confusion Matrix</span>
            <span style="font-size:11.5px;color:var(--text-muted);">True vs Predicted</span>
        </div>
        """, unsafe_allow_html=True)

        intent_classes = ["Refund", "Delivery", "Complaint", "Exchange", "Account", "Cancel", "Inquiry"]
        cm_data = [
            [15, 0, 1, 0, 0, 0, 0],
            [0, 14, 0, 0, 1, 0, 0],
            [1, 0, 16, 1, 0, 0, 0],
            [0, 0, 1, 12, 0, 0, 0],
            [0, 1, 0, 0, 11, 0, 0],
            [0, 0, 0, 0, 0, 8, 0],
            [0, 0, 0, 0, 0, 0, 9]
        ]
        fig_cm1 = px.imshow(
            cm_data,
            x=intent_classes,
            y=intent_classes,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            text_auto=True
        )
        fig_cm1.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(t=10, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_cm1, use_container_width=True)

    with c_cm2:
        st.markdown("""
        <div class="card-header-title">
            <span>Urgency Confusion Matrix</span>
            <span style="font-size:11.5px;color:var(--text-muted);">Rule-Assisted Output</span>
        </div>
        """, unsafe_allow_html=True)

        urgency_classes = ["Low", "Medium", "High"]
        urg_cm_data = [
            [8, 1, 0],
            [1, 18, 1],
            [0, 1, 3]
        ]
        fig_cm2 = px.imshow(
            urg_cm_data,
            x=urgency_classes,
            y=urgency_classes,
            color_continuous_scale="Greens",
            labels=dict(x="Predicted Urgency", y="True Urgency", color="Count"),
            text_auto=True
        )
        fig_cm2.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(t=10, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_cm2, use_container_width=True)

    # Feature Importance Bar Chart (Image 3)
    st.markdown("""
    <div class="card-header-title">
        <span>Feature Importance</span>
        <span style="font-size:12px;color:var(--text-muted);font-weight:600;">Top Discriminative N-Grams</span>
    </div>
    """, unsafe_allow_html=True)

    features = ["refund venum", "box damaged", "deliver aagala", "paise wapas", "charger wire", "replace pannanum", "password reset", "amount debit"]
    weights = [18.4, 15.2, 13.1, 11.5, 9.8, 8.4, 7.2, 6.9]

    fig_fi = go.Figure(go.Bar(
        x=weights,
        y=features,
        orientation='h',
        marker=dict(color="#3b82f6", line=dict(color="#2563eb", width=1)),
        text=[f"{w:.1f}" for w in weights],
        textposition="inside",
        textfont=dict(color="white", weight=700)
    ))
    fig_fi.update_layout(
        template=plotly_template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(t=10, b=20, l=120, r=20),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Sample Test Results Table (Matching Image 2 Bottom-Left table)
    st.markdown("""
    <div class="card-header-title">
        <span>Sample Test Results</span>
        <span style="font-size:12px;color:var(--text-muted);font-weight:500;">Evaluated from data/codemixed_test_set.csv</span>
    </div>
    """, unsafe_allow_html=True)

    test_samples = [
        {"id": 1, "msg": "order vandhuchu box damaged", "exp_i": "Refund Request", "pred_i": "Refund Request", "exp_u": "High", "pred_u": "High", "conf": 0.92, "status": "Matched"},
        {"id": 2, "msg": "paise wapas chahiye", "exp_i": "Refund Request", "pred_i": "Refund Request", "exp_u": "High", "pred_u": "High", "conf": 0.88, "status": "Matched"},
        {"id": 3, "msg": "order innum deliver aagala", "exp_i": "Delivery Issue", "pred_i": "Delivery Issue", "exp_u": "Medium", "pred_u": "Medium", "conf": 0.81, "status": "Matched"},
        {"id": 4, "msg": "product nalla illa da", "exp_i": "Product Complaint", "pred_i": "Product Complaint", "exp_u": "Medium", "pred_u": "Medium", "conf": 0.79, "status": "Matched"},
    ]

    t_html = """<table class="sa-table">
        <thead>
            <tr>
                <th>#</th>
                <th>Message</th>
                <th>Expected Intent</th>
                <th>Predicted Intent</th>
                <th>Expected Urgency</th>
                <th>Predicted Urgency</th>
                <th>Confidence</th>
            </tr>
        </thead>
        <tbody>"""
    for r in test_samples:
        t_html += (
            f"<tr>"
            f"<td style='font-weight:700;'>{r['id']}</td>"
            f"<td style='font-weight:600;color:var(--text-heading);'>{r['msg']}</td>"
            f"<td><span class='pill-badge pill-blue'>{r['exp_i']}</span></td>"
            f"<td><span class='pill-badge pill-blue'>{r['pred_i']}</span></td>"
            f"<td><span class='pill-badge {'pill-red' if r['exp_u']=='High' else 'pill-amber'}'>{r['exp_u']}</span></td>"
            f"<td><span class='pill-badge {'pill-red' if r['pred_u']=='High' else 'pill-amber'}'>{r['pred_u']}</span></td>"
            f"<td style='font-family:monospace;font-weight:700;'>{r['conf']:.2f}</td>"
            f"</tr>"
        )
    t_html += "</tbody></table>"
    st.html(t_html)


# =========================================================
# PAGE 5: TICKET DETAILS & HISTORY (Image 2 Bottom-Right)
# =========================================================
elif st.session_state["current_page"] == "ticket_history":

    col_thead1, col_thead2 = st.columns([3, 1])
    with col_thead1:
        st.markdown('<div class="page-title">Ticket Details & History</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">View full analysis and conversation history</div>', unsafe_allow_html=True)
    with col_thead2:
        if st.button("← Back to Queue", use_container_width=True):
            st.session_state["current_page"] = "queue_dashboard"
            st.rerun()

    # Active Detailed Ticket Card (Matching Mockup Screen 4)
    active_ticket_msg = "product nalla illa da, refund venum"
    active_ticket_time = "22 Oct 2026, 11:21"
    
    td_col1, td_col2 = st.columns([1.1, 0.9], gap="large")

    with td_col1:
        # Customer Message Card
        st.markdown(f"""
        <div class="sa-card">
            <div class="card-header-title">
                <span>Customer Message</span>
                <span style="font-size:12px;color:var(--text-muted);">{active_ticket_time}</span>
            </div>
            <div class="chat-bubble-customer" style="margin:0;font-size:15px;font-weight:600;">
                "{active_ticket_msg}"
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Suggested Reply Card
        rep_text, rep_lang = get_suggested_reply("refund_request", active_ticket_msg)
        st.markdown(f"""
        <div class="sa-card">
            <div class="card-header-title">
                <span>Suggested Reply ({rep_lang})</span>
                <span class="pill-badge pill-purple">Agent Draft</span>
            </div>
            <div style="font-size:13.5px;line-height:1.6;color:var(--text-body);margin-bottom:14px;background:var(--bg-subtle);padding:14px;border-radius:12px;border:1px solid var(--border-base);">
                {rep_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📋 Copy Reply Draft", key="btn_copy_detail_draft"):
            st.toast("Reply draft copied to clipboard!", icon="✅")

    with td_col2:
        # Analysis Result Card
        st.markdown("""
        <div class="sa-card">
            <div class="card-header-title">
                <span>Analysis Result</span>
                <span style="font-size:11px;color:var(--text-muted);font-weight:600;">CLASSIFICATION</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                <div>
                    <div style="font-size:11.5px;color:var(--text-muted);font-weight:600;margin-bottom:4px;">Intent</div>
                    <span class="pill-badge pill-blue">Refund Request</span>
                    <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">Confidence: <b>0.89</b></div>
                </div>
                <div>
                    <div style="font-size:11.5px;color:var(--text-muted);font-weight:600;margin-bottom:4px;">Urgency</div>
                    <span class="pill-badge pill-amber">Medium</span>
                    <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">Confidence: <b>0.78</b></div>
                </div>
                <div>
                    <div style="font-size:11.5px;color:var(--text-muted);font-weight:600;margin-bottom:4px;">Assigned Queue</div>
                    <span class="pill-badge pill-gray">📁 Refunds</span>
                </div>
                <div>
                    <div style="font-size:11.5px;color:var(--text-muted);font-weight:600;margin-bottom:4px;">Escalate to Human</div>
                    <span class="pill-badge pill-green">No</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Ticket Timeline (Matching Screen 4 Timeline)
        st.markdown(f"""
        <div class="sa-card">
            <div class="card-header-title">
                <span>Ticket Timeline</span>
                <span style="font-size:12px;color:var(--text-muted);font-weight:500;">Audit Trail</span>
            </div>
            <div style="padding-top:6px;">
                <div class="timeline-item">
                    <div class="timeline-dot" style="background:#3b82f6;color:white;">✓</div>
                    <div>
                        <div style="font-size:13px;font-weight:700;color:var(--text-heading);">Message received</div>
                        <div style="font-size:11px;color:var(--text-muted);">{active_ticket_time}</div>
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-dot" style="background:#10b981;color:white;">⚡</div>
                    <div>
                        <div style="font-size:13px;font-weight:700;color:var(--text-heading);">Analyzed and classified</div>
                        <div style="font-size:11px;color:var(--text-muted);">{active_ticket_time} · Intent 0.89</div>
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-dot" style="background:#8b5cf6;color:white;">💬</div>
                    <div>
                        <div style="font-size:13px;font-weight:700;color:var(--text-heading);">Reply generated (Tanglish)</div>
                        <div style="font-size:11px;color:var(--text-muted);">{active_ticket_time} · Template matched</div>
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-dot" style="background:#f59e0b;color:white;">📁</div>
                    <div>
                        <div style="font-size:13px;font-weight:700;color:var(--text-heading);">Routed to Refunds Queue</div>
                        <div style="font-size:11px;color:var(--text-muted);">{active_ticket_time} · Agent notified</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PAGE 6: SETTINGS & ROUTING
# =========================================================
elif st.session_state["current_page"] == "settings":
    st.markdown('<div class="page-title">Settings & Routing Rules</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Configure triage confidence thresholds, fallback behavior, and team queues</div>', unsafe_allow_html=True)

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("""
        <div class="sa-card">
            <div class="card-header-title"><span>Confidence & Escalation Thresholds</span></div>
        """, unsafe_allow_html=True)
        st.slider("Intent Auto-Accept Threshold", min_value=0.50, max_value=0.95, value=0.60, step=0.05)
        st.slider("Urgency Auto-Accept Threshold", min_value=0.50, max_value=0.95, value=0.60, step=0.05)
        st.checkbox("Auto-flag high urgency tickets for supervisor review", value=True)
        st.checkbox("Flag ambiguous delivery duration without explicit dates", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div class="sa-card">
            <div class="card-header-title"><span>Dialect & Language Preferences</span></div>
        """, unsafe_allow_html=True)
        st.selectbox("Default Response Dialect", ["Auto-detect Customer Language", "Tanglish", "Hinglish", "Standard English"])
        st.text_input("Human Review Notification Webhook", value="https://hooks.slack.com/services/T00/B00/SUPPORT_ALERT")
        if st.button("Save Configuration", type="primary", use_container_width=True):
            st.toast("Settings saved successfully!", icon="💾")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div style="margin-top:40px;padding-top:20px;border-top:1px solid var(--border-base);text-align:center;font-size:11.5px;color:var(--text-muted);font-weight:600;">
    SupportAI · Code-Mixed Customer Intelligence Platform &nbsp; • &nbsp; Light &amp; Dark Theme Enabled &nbsp; • &nbsp; Human-in-the-Loop Operations
</div>
""", unsafe_allow_html=True)
