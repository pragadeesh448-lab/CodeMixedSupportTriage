"""Shared rule-assisted urgency logic for the support triage prototype."""

import re


HIGH_PAYMENT_PATTERNS = [
    r"\bdeducted twice\b",
    r"\bcharged twice\b",
    r"\bdouble debit\b",
    r"\bdouble charge\b",
    r"payment failed.{0,60}(money|amount|paise|paisa).{0,30}(deduct\\w*|debit\\w*|cut|kat)",
    r"charged an extra fee",
    r"charged extra",
    r"extra fee",
    r"(money|amount).{0,30}(deducted|debited) twice",
    r"paise kat gaye",
    r"paise kat gaye lekin",
    r"rendu thadava debit",
    r"extra money deduct",
    r"extra amount cut",
    r"amount debit aagiduchu",
    r"amount was debited",
    r"amount debit aachu but order place aagala",
]

HIGH_DELAY_PATTERNS = [
    r"\b3\s+days?\b",
    r"\bthree\s+days?\b",
    r"\bteen din\b",
    r"\bromba late\b",
    r"\bbahut late\b",
    r"\bromba naal\b",
    r"\blong time\b",
    r"\brefund innum varala\b",
    r"\brefund abhi tak nahi aaya\b",
    r"\brefund abhi tak nahi mila\b",
    r"\babhi tak.{0,30}refund.{0,30}nahi mila\b",
    r"\brefund nahi mila\b",
    r"\brefund nahi aaya\b",
    r"\brefund amount urgent\b",
    r"\brefund urgent\b",
    r"\bstill waiting for my refund\b",
    r"\bi am still waiting for my refund\b",
    r"\brefund has not arrived\b",
    r"\brefund hasn't arrived\b",
    r"\brefund not received\b",
    r"refund approve.{0,50}(not credited|not received|credit aagala|paise nahi mile)",
]

HIGH_ACCOUNT_PATTERNS = [
    r"\baccount (is )?(locked|lock|blocked)\b",
    r"\baccess is blocked\b",
    r"\baccount lock ho gaya\b",
    r"\baccount access is blocked\b",
]

LOW_ROUTINE_PATTERNS = [
    r"\bpassword reset\b",
    r"\bpassword marandhuten\b",
    r"\bpassword bhool gaya\b",
    r"\bforgot (my )?password\b",
    r"\bhelp me sign in\b",
    r"\bhelp me login\b",
    r"\bsign in to my account\b",
    r"\brefund status\b",
    r"\brefund process status\b",
    r"\brefund status check\b",
    r"\bstatus check\b",
    r"\bstatus update\b",
    r"\bwhere is my delivery\b",
    r"\border tracking status\b",
    r"\border ka track\b",
]


def _matches_any(text, patterns):
    return any(re.search(pattern, text) for pattern in patterns)


def predict_rule_urgency(message, predicted_intent, urgency_model, urgency_encoder):
    """Return rule-assisted urgency, using the ML model as a fallback."""
    text = re.sub(r"\s+", " ", str(message).lower()).strip()

    # High-risk signals take precedence over broad, routine intent patterns.
    if _matches_any(text, HIGH_PAYMENT_PATTERNS):
        return "high"
    if _matches_any(text, HIGH_DELAY_PATTERNS):
        return "high"
    if _matches_any(text, HIGH_ACCOUNT_PATTERNS):
        return "high"

    if _matches_any(text, LOW_ROUTINE_PATTERNS):
        return "low"

    # Ordinary refund/return/cancellation requests default to medium.
    if predicted_intent in {"return_request", "order_cancellation", "refund_request", "payment_issue", "account_support"}:
        return "medium"

    # Unclear delivery and other unmatched messages retain the ML estimate.
    probabilities = urgency_model.predict_proba([message])[0]
    index = probabilities.argmax()
    return str(urgency_encoder.inverse_transform([index])[0])


def is_ambiguous_delivery(message, predicted_intent):
    """Flag delivery messages that do not state a clear delay duration."""
    if predicted_intent != "delivery_issue":
        return False

    text = re.sub(r"\s+", " ", str(message).lower()).strip()
    explicit_delay_patterns = [
        r"\b3\s+days?\b",
        r"\bthree\s+days?\b",
        r"\bteen din\b",
        r"\bromba late\b",
        r"\bbahut late\b",
        r"\bromba naal\b",
        r"\blong time\b",
    ]
    return not _matches_any(text, explicit_delay_patterns)
