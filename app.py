import re
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from faker import Faker
from textblob import TextBlob


# =========================
# App Config
# =========================
st.set_page_config(page_title="Support Insights Dashboard", page_icon="📊", layout="wide")
APP_TITLE = "Support Insights Dashboard"
DEFAULT_CASE_COUNT = 500


# =========================
# ZoomInfo-like Salesforce Statuses (Tier 1 style)
# =========================
STATUSES = [
    "New",
    "Awaiting Response",
    "Responded",
    "Reopened",
    "Working",
    "In Progress",
    "Resolved",
]

OPEN_STATUSES = {
    "New",
    "Awaiting Response",
    "Responded",
    "Reopened",
    "Working",
    "In Progress",
}
CLOSED_STATUSES = {"Resolved"}

# Origin (mostly Email)
ORIGINS = ["Email"]


# =========================
# Taxonomy / Options (platform-agnostic, realistic enterprise support)
# =========================
CATEGORIES = [
    "Admin / User Management",
    "Credits / Export Limits",
    "Intent Data Confusion",
    "Contact Data Accuracy",
    "Platform Performance",
    "Integration Issues",
    "Login / Access Issues",
    "Browser Extension Issues",
    "Billing / Subscription",
    "Feature Clarification",
]

# Tier-1 note: Priority/Severity often not present.
# We keep a generic "Urgency" field optional only if you want later.
# For now we do NOT generate Priority/Severity to match your Tier-1 reality.

RESOLUTION_TYPES = [
    "How-To Guidance",
    "Bug Fix",
    "Data Correction",
    "Access Restored",
    "Workaround Provided",
    "Escalated to Eng",
]

# Sentiment shaping language (to keep analytics realistic)
NEGATIVE_PHRASES = [
    "This is extremely frustrating",
    "We've been blocked for days",
    "This is impacting our team heavily",
    "Still not resolved after multiple attempts",
    "We are losing time and productivity",
    "This keeps happening repeatedly",
    "We need an urgent fix",
    "This is unacceptable for an enterprise workflow",
]
NEUTRAL_PHRASES = [
    "Requesting clarification on expected behavior",
    "Need help understanding how this should work",
    "We have a question about configuration",
    "Can you confirm the correct workflow?",
    "Seeking guidance on best practices",
]
POSITIVE_PHRASES = [
    "Thanks, that helped",
    "This looks good now",
    "Appreciate the quick turnaround",
    "Resolved on our end",
    "Everything is working as expected",
]

# Accounts to repeat (enterprise feel)
ENTERPRISE_ACCOUNTS = [
    "Cisco",
    "Intuit",
    "Salesforce",
    "Paychex",
    "Cintas",
    "Adobe",
    "Oracle",
    "VMware",
    "ServiceNow",
    "Accenture",
    "Deloitte",
    "PwC",
    "KPMG",
    "Infosys",
    "TCS",
    "Wipro",
]

CATEGORY_TO_KEYWORDS = {
    "Admin / User Management": ["admin", "user role", "permission", "seat", "provision", "SSO", "SCIM", "onboarding", "offboarding"],
    "Credits / Export Limits": ["credits", "export", "limit", "bulk", "download", "quota"],
    "Intent Data Confusion": ["intent", "topic", "signal", "surge", "intent filter", "topic selection"],
    "Contact Data Accuracy": ["email accuracy", "bounce", "wrong title", "bad contact", "missing phone", "outdated"],
    "Platform Performance": ["slow", "lag", "timeout", "loading", "error", "performance"],
    "Integration Issues": ["Salesforce", "CRM", "API", "webhook", "integration", "sync", "connector"],
    "Login / Access Issues": ["login", "password", "MFA", "2FA", "SSO", "access denied", "locked out"],
    "Browser Extension Issues": ["extension", "chrome", "edge", "plugin", "toolbar", "not showing"],
    "Billing / Subscription": ["invoice", "billing", "subscription", "renewal", "contract", "seat count"],
    "Feature Clarification": ["how to", "best practice", "workflow", "does this support", "can we", "where do I"],
}


# =========================
# Session State
# =========================
def init_state():
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "clean_df" not in st.session_state:
        st.session_state.clean_df = None
    if "pii_metrics" not in st.session_state:
        st.session_state.pii_metrics = None
    if "analysis" not in st.session_state:
        st.session_state.analysis = None


init_state()


# =========================
# Helpers
# =========================
def weighted_choice(items, weights, rng: np.random.Generator):
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return items[int(rng.choice(len(items), p=weights))]


def make_case_text(category: str, rng: np.random.Generator):
    kws = CATEGORY_TO_KEYWORDS.get(category, ["help", "question", "issue"])
    base = f"{rng.choice(kws)} issue: {rng.choice(kws)} not working as expected."

    # intentionally skew negative-ish for realism
    bucket = weighted_choice(["negative", "neutral", "positive"], [0.42, 0.38, 0.20], rng)

    if bucket == "negative":
        addon = rng.choice(NEGATIVE_PHRASES)
    elif bucket == "positive":
        addon = rng.choice(POSITIVE_PHRASES)
    else:
        addon = rng.choice(NEUTRAL_PHRASES)

    details = [
        "We tried clearing cache and cookies but the behavior persists.",
        "We see the issue across multiple users in our org.",
        "It works for some users but fails for others.",
        "We validated permissions and still see the same error.",
        "This appears after a recent change in settings.",
        "Please confirm if this is expected behavior.",
    ]
    return bucket, f"{base} {addon}. {rng.choice(details)}"


def generate_synthetic_cases(n: int, seed: int = 42) -> pd.DataFrame:
    fake = Faker()
    Faker.seed(seed)
    rng = np.random.default_rng(seed)

    # Repeat-account weighting: enterprise accounts appear more often
    accounts = ENTERPRISE_ACCOUNTS + [fake.company() for _ in range(40)]
    weights = np.ones(len(accounts))
    for i, name in enumerate(accounts):
        if name in ENTERPRISE_ACCOUNTS:
            weights[i] = 3.5
    weights = weights / weights.sum()

    start_date = datetime.now() - timedelta(days=30)
    rows = []

    for _ in range(n):
        case_id = f"CASE-{uuid.uuid4().hex[:10].upper()}"
        category = str(rng.choice(CATEGORIES))

        # Status distribution: ensure realistic open vs resolved
        status = weighted_choice(
            STATUSES,
            # more open cases than resolved for realism in an active queue
            [0.10, 0.22, 0.14, 0.06, 0.22, 0.18, 0.08],
            rng,
        )

        # Escalation chance slightly higher for performance/integration issues (still subtle)
        base_escalation = 0.22
        if category in ["Platform Performance", "Integration Issues"]:
            base_escalation = 0.28
        escalation = bool(rng.choice([0, 1], p=[1 - base_escalation, base_escalation]))

        # Resolution only if resolved
        resolution = str(rng.choice(RESOLUTION_TYPES)) if status in CLOSED_STATUSES else "—"

        account_name = accounts[int(rng.choice(len(accounts), p=weights))]
        contact_name = fake.name()
        email = fake.email()
        phone = fake.phone_number()

        # Subject feels Salesforce-ish and short
        keyword = rng.choice(CATEGORY_TO_KEYWORDS[category])
        subject = f"{category} - {keyword} issue"

        synth_bucket, description = make_case_text(category, rng)

        created_at = start_date + timedelta(minutes=int(rng.integers(0, 30 * 24 * 60)))
        updated_at = created_at + timedelta(hours=int(rng.integers(1, 72)))

        origin = "Email"

        rows.append(
            {
                "Case_ID": case_id,
                "Created_At": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "Updated_At": updated_at.strftime("%Y-%m-%d %H:%M:%S"),
                "Origin": origin,
                "Account_Name": account_name,
                "Contact_Name": contact_name,
                "Email": email,
                "Phone": phone,
                "Subject": subject,
                "Description": description,
                "Category": category,
                "Status": status,
                "Escalation_Flag": escalation,
                "Resolution_Type": resolution,
                # helpful for future expansions (kept generic)
                "Region": weighted_choice(["NA", "EMEA", "APAC", "LATAM"], [0.45, 0.25, 0.20, 0.10], rng),
                "Synthetic_Sentiment_Label": synth_bucket,
            }
        )

    return pd.DataFrame(rows)


# =========================
# PII Scrubbing (Token Replace)
# =========================
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\-\s\(\)]{7,}\d)")
NAME_IN_TEXT_RE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})\b")
POSTAL_RE = re.compile(r"\b\d{5,6}\b")
ADDRESS_HINT_RE = re.compile(r"\b(street|st\.|road|rd\.|avenue|ave\.|lane|ln\.|block|building|apartment|apt\.|zip|postal)\b", re.IGNORECASE)


def scrub_pii(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned = df.copy()
    metrics = {
        "emails_replaced": 0,
        "phones_replaced": 0,
        "names_replaced_in_text": 0,
        "postal_codes_replaced": 0,
        "address_hints_flagged": 0,
        "contact_names_replaced_field": 0,
    }

    # Field-level replacement
    if "Email" in cleaned.columns:
        metrics["emails_replaced"] = int(cleaned["Email"].notna().sum())
        cleaned["Email"] = "[EMAIL_REDACTED]"

    if "Phone" in cleaned.columns:
        metrics["phones_replaced"] = int(cleaned["Phone"].notna().sum())
        cleaned["Phone"] = "[PHONE_REDACTED]"

    if "Contact_Name" in cleaned.columns:
        metrics["contact_names_replaced_field"] = int(cleaned["Contact_Name"].notna().sum())
        cleaned["Contact_Name"] = "[NAME_REDACTED]"

    def _scrub_text(text: str) -> str:
        if not isinstance(text, str):
            return text

        metrics["emails_replaced"] += len(EMAIL_RE.findall(text))
        metrics["phones_replaced"] += len(PHONE_RE.findall(text))
        metrics["names_replaced_in_text"] += len(NAME_IN_TEXT_RE.findall(text))
        metrics["postal_codes_replaced"] += len(POSTAL_RE.findall(text))
        metrics["address_hints_flagged"] += 1 if ADDRESS_HINT_RE.search(text) else 0

        text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
        text = PHONE_RE.sub("[PHONE_REDACTED]", text)
        text = POSTAL_RE.sub("[POSTAL_REDACTED]", text)
        text = NAME_IN_TEXT_RE.sub("[NAME_REDACTED]", text)
        return text

    for col in ["Subject", "Description"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].apply(_scrub_text)

    return cleaned, metrics


# =========================
# Analysis
# =========================
def compute_sentiment(subject: str, description: str) -> float:
    text = f"{subject} {description}".strip()
    try:
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


def sentiment_bucket(score: float) -> str:
    if score <= -0.10:
        return "Negative"
    if score >= 0.10:
        return "Positive"
    return "Neutral"


def run_analysis(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["Sentiment_Score"] = work.apply(
        lambda r: compute_sentiment(r.get("Subject", ""), r.get("Description", "")),
        axis=1,
    )
    work["Sentiment_Label"] = work["Sentiment_Score"].apply(sentiment_bucket)

    # Open/Closed counts
    work["Open_or_Closed"] = work["Status"].apply(lambda s: "Open" if s in OPEN_STATUSES else "Closed")

    kpis = {
        "total_cases": int(len(work)),
        "open_cases": int((work["Open_or_Closed"] == "Open").sum()),
        "resolved_cases": int((work["Status"] == "Resolved").sum()),
        "escalations": int(work["Escalation_Flag"].sum()),
        "negative_cases": int((work["Sentiment_Label"] == "Negative").sum()),
        "repeat_accounts": int((work["Account_Name"].value_counts() > 1).sum()),
    }

    dist = {
        "status": work["Status"].value_counts().reindex(STATUSES).fillna(0).astype(int),
        "open_closed": work["Open_or_Closed"].value_counts().reindex(["Open", "Closed"]).fillna(0).astype(int),
        "category": work["Category"].value_counts().sort_values(ascending=False),
        "sentiment": work["Sentiment_Label"].value_counts().reindex(["Negative", "Neutral", "Positive"]).fillna(0).astype(int),
        "resolution": work["Resolution_Type"].value_counts().sort_values(ascending=False),
        "escalations_by_category": work.groupby("Category")["Escalation_Flag"].sum().sort_values(ascending=False),
        "negative_by_category": work[work["Sentiment_Label"] == "Negative"]["Category"].value_counts().sort_values(ascending=False),
        "top_repeat_accounts": work["Account_Name"].value_counts()[lambda s: s > 1].head(10),
        "origin": work["Origin"].value_counts().sort_values(ascending=False),
    }

    return {"df_scored": work, "kpis": kpis, "dist": dist}


# =========================
# Plotting
# =========================
def plot_bar(series: pd.Series, title: str, xlabel: str = "", ylabel: str = "Count"):
    fig, ax = plt.subplots()
    series = series.copy()
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# =========================
# UI
# =========================
st.title(APP_TITLE)
st.caption("A local, reusable dashboard for generating support cases, scrubbing PII, and analyzing friction signals.")

with st.sidebar:
    st.header("Controls")
    case_count = st.number_input("Number of synthetic cases", min_value=100, max_value=3000, value=DEFAULT_CASE_COUNT, step=100)
    seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)

    st.divider()
    st.subheader("Step 1 — Generate")
    if st.button("Generate Synthetic Cases", use_container_width=True):
        st.session_state.raw_df = generate_synthetic_cases(int(case_count), int(seed))
        st.session_state.clean_df = None
        st.session_state.pii_metrics = None
        st.session_state.analysis = None
        st.success(f"Generated {len(st.session_state.raw_df)} cases.")

    st.subheader("Step 2 — Scrub PII")
    if st.button("Scrub PII (Token Replace)", use_container_width=True, disabled=(st.session_state.raw_df is None)):
        st.session_state.clean_df, st.session_state.pii_metrics = scrub_pii(st.session_state.raw_df)
        st.session_state.analysis = None
        st.success("PII scrubbing complete.")

    st.subheader("Step 3 — Analyze")
    if st.button("Run Analysis", use_container_width=True, disabled=(st.session_state.clean_df is None)):
        st.session_state.analysis = run_analysis(st.session_state.clean_df)
        st.success("Analysis complete.")

    st.divider()
    if st.session_state.clean_df is not None:
        csv_bytes = st.session_state.clean_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned CSV",
            data=csv_bytes,
            file_name=f"support_cases_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# Main layout
col_a, col_b = st.columns([1, 1], gap="large")

with col_a:
    st.subheader("Dataset Status")
    if st.session_state.raw_df is None:
        st.info("No dataset yet. Use **Step 1 — Generate** in the sidebar.")
    else:
        st.write("**Raw dataset preview (first 10 rows):**")
        st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)

with col_b:
    st.subheader("PII Scrubbing Metrics")
    if st.session_state.pii_metrics is None:
        st.info("PII metrics will appear after you run **Step 2 — Scrub PII**.")
    else:
        m = st.session_state.pii_metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Emails replaced", f"{m['emails_replaced']}")
        m2.metric("Phones replaced", f"{m['phones_replaced']}")
        m3.metric("Contact names replaced", f"{m['contact_names_replaced_field']}")

        m4, m5, m6 = st.columns(3)
        m4.metric("Names replaced (text)", f"{m['names_replaced_in_text']}")
        m5.metric("Postal codes replaced", f"{m['postal_codes_replaced']}")
        m6.metric("Address hints flagged", f"{m['address_hints_flagged']}")

        st.write("**Cleaned dataset preview (first 10 rows):**")
        st.dataframe(st.session_state.clean_df.head(10), use_container_width=True)


st.divider()
st.subheader("Insights")

if st.session_state.analysis is None:
    st.info("Run **Step 3 — Analyze** to view insights and charts.")
else:
    analysis = st.session_state.analysis
    k = analysis["kpis"]
    d = analysis["dist"]

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total cases", f"{k['total_cases']}")
    k2.metric("Open cases", f"{k['open_cases']}")
    k3.metric("Resolved", f"{k['resolved_cases']}")
    k4.metric("Escalations", f"{k['escalations']}")
    k5.metric("Negative cases", f"{k['negative_cases']}")
    k6.metric("Repeat accounts", f"{k['repeat_accounts']}")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.pyplot(plot_bar(d["open_closed"], "Open vs Closed"))
        st.pyplot(plot_bar(d["sentiment"], "Sentiment Distribution"))

    with c2:
        st.pyplot(plot_bar(d["status"], "Case Status Distribution"))
        st.pyplot(plot_bar(d["category"].head(10), "Top Categories by Volume"))

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.pyplot(plot_bar(d["escalations_by_category"].head(10), "Escalations by Category (Top 10)"))

    with c4:
        neg = d["negative_by_category"]
        if len(neg) == 0:
            st.info("No negative sentiment detected with current dataset.")
        else:
            st.pyplot(plot_bar(neg.head(10), "Negative Sentiment by Category (Top 10)"))

    c5, c6 = st.columns(2, gap="large")
    with c5:
        rep = d["top_repeat_accounts"]
        if len(rep) == 0:
            st.info("No repeat accounts in this dataset (unexpected).")
        else:
            st.pyplot(plot_bar(rep, "Top Repeat Accounts"))

    with c6:
        st.pyplot(plot_bar(d["origin"], "Case Origin"))

    st.divider()
    st.subheader("Scored Dataset (Preview)")
    st.dataframe(analysis["df_scored"].head(20), use_container_width=True)