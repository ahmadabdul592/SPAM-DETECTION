import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from urllib.parse import urlparse
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* gradient header */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero h1 { color: #e94560; font-size: 2.4rem; margin: 0; font-weight: 700; }
.hero p  { color: #a8b2d8; font-size: 1rem; margin-top: .5rem; }

/* metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-title { color: #a8b2d8; font-size: .8rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #e94560; font-size: 2rem; font-weight: 700; }

/* result banner */
.result-spam {
    background: linear-gradient(135deg, #e94560, #c0392b);
    border-radius: 12px; padding: 1.5rem; text-align: center;
    color: white; font-size: 1.6rem; font-weight: 700;
    box-shadow: 0 4px 20px rgba(233,69,96,0.4);
    animation: pulse 1.5s infinite;
}
.result-ham {
    background: linear-gradient(135deg, #2ecc71, #27ae60);
    border-radius: 12px; padding: 1.5rem; text-align: center;
    color: white; font-size: 1.6rem; font-weight: 700;
    box-shadow: 0 4px 20px rgba(46,204,113,0.4);
}
@keyframes pulse {
  0%,100% { box-shadow: 0 4px 20px rgba(233,69,96,0.4); }
  50%      { box-shadow: 0 4px 40px rgba(233,69,96,0.8); }
}

/* URL risk badge */
.risk-low  { background:#2ecc71; color:#fff; border-radius:8px; padding:.3rem .8rem; font-weight:600; }
.risk-med  { background:#f39c12; color:#fff; border-radius:8px; padding:.3rem .8rem; font-weight:600; }
.risk-high { background:#e94560; color:#fff; border-radius:8px; padding:.3rem .8rem; font-weight:600; }

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] .css-1d391kg { color: #a8b2d8; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #16213e; border-radius: 8px; color: #a8b2d8;
    padding: .5rem 1.2rem; font-weight: 600;
}
.stTabs [aria-selected="true"] { background: #e94560 !important; color: white !important; }

/* text area */
.stTextArea textarea {
    background: #16213e; color: #e8e8e8; border: 1px solid #0f3460;
    border-radius: 10px; font-size: 1rem;
}

/* button */
.stButton > button {
    background: linear-gradient(135deg, #e94560, #c0392b);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; font-size: 1rem; padding: .6rem 2rem;
    width: 100%; transition: all .2s;
}
.stButton > button:hover { transform: translateY(-2px); opacity: .9; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  DATA  (built-in mini-corpus so no CSV needed)
# ─────────────────────────────────────────────
SPAM_SAMPLES = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
    "Congratulations! You've been selected to win a free iPhone. Click now!",
    "URGENT! Your account has been compromised. Verify now at http://secure-login.xyz",
    "Win money now!!! Click http://free-money-bonus.xyz to claim your prize.",
    "You have won $1000 cash prize! Call now to claim your reward.",
    "FREE MESSAGE: Congrats, ur awarded either £500 of CD vouchers or 125gift.",
    "Todays Voda numbers ending with 7634 are selected to receive a £350 award.",
    "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575.",
    "IMPORTANT! Your Mobile number won a £1,000 Bonus Caller Prize on 02/12/04.",
    "Claim your prize by replying to this message with your details.",
    "You are a WINNER! Call 09061743806 from a landline.",
    "Had your mobile 11 months or more? You are entitled to update your FREE handset.",
    "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set",
    "WINNER! You've been randomly selected to receive a gift. Respond now.",
    "Free ringtone offer! Text RING to 88600 to get your free ringtone.",
    "Urgent! Your account will be suspended. Click to verify now.",
    "You have a secret admirer. Text GIRL to 85068 for details.",
    "Cheap flights! Book now at http://discountflights-verify.com",
    "CLAIM your £150 Argos voucher now. Call 08717895698.",
]

HAM_SAMPLES = [
    "Hey, are we still meeting today at noon?",
    "Ok lar... Joking wif u oni...",
    "Nah I don't think he goes to usf, he lives around here though",
    "Go until jurong point, crazy.. Available only in bugis n great world la e buffet...",
    "U dun say so early hor... U c already then say...",
    "I'm at work now, I'll call you after lunch.",
    "Good morning! Hope you have a great day.",
    "Can you pick up some milk on the way home?",
    "The meeting has been rescheduled to 3 PM tomorrow.",
    "Happy birthday! Hope you have a wonderful day.",
    "Thank you for the gift, it was really thoughtful.",
    "I'll be there in 10 minutes, just finishing up.",
    "Did you see the game last night? Amazing finish!",
    "Let me know when you arrive and I'll come down.",
    "The report is ready for your review.",
    "Looking forward to seeing you this weekend!",
    "Can we reschedule our call to Friday?",
    "I've attached the documents you requested.",
    "No worries, take your time with the proposal.",
    "Just checking in — how are you doing?",
    "Great work on the presentation today!",
    "The kids are asking when grandma is visiting.",
    "I'll send you the details once I get home.",
    "Your package has been delivered to the front door.",
    "Don't forget we have dinner at 7 tonight.",
    "Class is cancelled tomorrow due to the weather.",
    "See you at the gym at 6am?",
    "I finished reading that book you recommended.",
    "The plumber is coming Wednesday between 10-12.",
    "Hope the interview went well!",
]

# Build DataFrame
df = pd.DataFrame(
    [(1, t) for t in SPAM_SAMPLES] + [(0, t) for t in HAM_SAMPLES],
    columns=["label", "message"],
)

# ─────────────────────────────────────────────
#  MODEL  (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model():
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(df["message"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    return model, vectorizer, cm, report, fpr, tpr, roc_auc, prec, rec


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS  (same logic as notebook)
# ─────────────────────────────────────────────
def extract_url(text):
    urls = re.findall(r"(https?://\S+)", text)
    return urls[0] if urls else None


def remove_urls(text):
    return re.sub(r"http[s]?://\S+", "", text)


TRUSTED_DOMAINS = ["google.com", "github.com", "microsoft.com", "openai.com", "wikipedia.org"]
SUSPICIOUS_WORDS = ["login", "verify", "secure", "bank", "update", "free", "bonus", "win"]


def url_risk_score(url):
    score = 0
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if any(td in domain for td in TRUSTED_DOMAINS):
        return 0
    if len(url) > 75:
        score += 1
    if "@" in url or "%" in url:
        score += 2
    if url.count("-") > 2:
        score += 1
    if not url.startswith("https"):
        score += 1
    for word in SUSPICIOUS_WORDS:
        if word in url.lower():
            score += 1
    return score


def url_risk_details(url):
    flags = []
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if any(td in domain for td in TRUSTED_DOMAINS):
        return 0, ["✅ Trusted domain — no risk"]
    if len(url) > 75:
        flags.append("⚠️ Very long URL (>75 chars)")
    if "@" in url or "%" in url:
        flags.append("🚨 Contains @ or % encoding")
    if url.count("-") > 2:
        flags.append("⚠️ Multiple hyphens in URL")
    if not url.startswith("https"):
        flags.append("⚠️ Not using HTTPS")
    for word in SUSPICIOUS_WORDS:
        if word in url.lower():
            flags.append(f"🚨 Suspicious keyword: '{word}'")
    score = url_risk_score(url)
    if not flags:
        flags.append("✅ No suspicious patterns found")
    return score, flags


def final_prediction(message):
    url = extract_url(message)
    cleaned = remove_urls(message)
    model, vectorizer, *_ = train_model()
    vec = vectorizer.transform([cleaned])
    text_pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][1]
    risk = url_risk_score(url) if url else 0
    if text_pred == 1 or risk >= 3:
        verdict = "SPAM"
    else:
        verdict = "NOT SPAM"
    return verdict, text_pred, proba, url, risk


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#e94560;text-align:center;'>🛡️ Spam Shield</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#a8b2d8;font-size:.85rem;text-align:center;'>Hybrid ML + URL Risk Engine</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("<p style='color:#a8b2d8;font-weight:600;'>📊 Model Info</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a8b2d8;font-size:.82rem;'>• Algorithm: Logistic Regression<br>• Features: TF-IDF (5000)<br>• URL Risk Scoring<br>• Hybrid final decision</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='color:#a8b2d8;font-weight:600;'>🔍 Detection Rules</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a8b2d8;font-size:.82rem;'>• Text model predicts SPAM → SPAM<br>• URL risk score ≥ 3 → SPAM<br>• Otherwise → NOT SPAM</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown(
        "<p style='color:#a8b2d8;font-size:.75rem;text-align:center;'>Built with ❤️ using Streamlit</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  LOAD MODEL (with spinner)
# ─────────────────────────────────────────────
with st.spinner("🔄 Training hybrid spam detection model…"):
    model, vectorizer, cm, report, fpr, tpr, roc_auc, prec, rec = train_model()

# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <h1>🛡️ Hybrid Spam Detector</h1>
  <p>Combining NLP Text Classification with URL Risk Analysis for superior spam detection</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  TOP METRICS ROW
# ─────────────────────────────────────────────
acc = report["accuracy"]
prec_val = report["weighted avg"]["precision"]
rec_val = report["weighted avg"]["recall"]
f1_val = report["weighted avg"]["f1-score"]

c1, c2, c3, c4 = st.columns(4)
for col, label, val, fmt in zip(
    [c1, c2, c3, c4],
    ["Accuracy", "Precision", "Recall", "F1-Score"],
    [acc, prec_val, rec_val, f1_val],
    ["{:.1%}", "{:.1%}", "{:.1%}", "{:.1%}"],
):
    col.markdown(
        f"""<div class="metric-card">
  <div class="metric-title">{label}</div>
  <div class="metric-value">{fmt.format(val)}</div>
</div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Spam Detector", "📊 Model Analytics", "🧪 Batch Tester"])

# ══════════════════════════════════════════════
#  TAB 1 — DETECTOR
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### ✉️ Analyze a Message")
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        user_msg = st.text_area(
            "Paste your message below:",
            height=160,
            placeholder="e.g. URGENT! Verify your account https://secure-bank-update.xyz",
            key="msg_input",
        )

        # Example buttons
        st.markdown("<p style='color:#a8b2d8;font-size:.85rem;margin-bottom:4px;'>Quick examples:</p>", unsafe_allow_html=True)
        ex_col1, ex_col2 = st.columns(2)
        examples = {
            "🚨 Spam #1": "Win money now!!! Click http://free-money-login.xyz to claim your prize!",
            "✅ Ham #1": "Hey, are we still meeting today at noon?",
            "🚨 Spam #2": "URGENT! Verify your account https://secure-bank-update-login.xyz",
            "✅ Ham #2": "The meeting has been rescheduled to 3 PM tomorrow.",
        }
        for i, (label, txt) in enumerate(examples.items()):
            btn_col = ex_col1 if i % 2 == 0 else ex_col2
            if btn_col.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state["msg_input"] = txt
                st.rerun()

        analyze_btn = st.button("🔍 Analyze Message", use_container_width=True, key="analyze")

    with col_result:
        if analyze_btn and user_msg.strip():
            with st.spinner("Analyzing…"):
                time.sleep(0.4)
                verdict, text_pred, proba, url, risk = final_prediction(user_msg)

            # Verdict banner
            if verdict == "SPAM":
                st.markdown(f'<div class="result-spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-ham">✅ NOT SPAM — Safe</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence gauge
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=round(proba * 100, 1),
                    title={"text": "Spam Probability %", "font": {"color": "#a8b2d8", "size": 14}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#a8b2d8"},
                        "bar": {"color": "#e94560" if proba > 0.5 else "#2ecc71"},
                        "bgcolor": "#16213e",
                        "bordercolor": "#0f3460",
                        "steps": [
                            {"range": [0, 40], "color": "#1a2a1a"},
                            {"range": [40, 70], "color": "#2a2a1a"},
                            {"range": [70, 100], "color": "#2a1a1a"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 2},
                            "thickness": 0.8,
                            "value": 50,
                        },
                    },
                    number={"font": {"color": "#e8e8e8", "size": 28}, "suffix": "%"},
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=220,
                margin=dict(t=30, b=0, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Details
            d1, d2 = st.columns(2)
            d1.metric("Text Model", "🔴 SPAM" if text_pred == 1 else "🟢 HAM")
            d2.metric("URL Risk Score", f"{risk}/8" if url else "N/A")

            # URL details
            if url:
                st.markdown("**🔗 URL Analysis:**")
                score, flags = url_risk_details(url)
                risk_label = (
                    f'<span class="risk-high">HIGH RISK ({score})</span>'
                    if score >= 3
                    else f'<span class="risk-med">MEDIUM ({score})</span>'
                    if score >= 1
                    else f'<span class="risk-low">LOW RISK ({score})</span>'
                )
                st.markdown(f"`{url}`  →  {risk_label}", unsafe_allow_html=True)
                for flag in flags:
                    st.markdown(f"&nbsp;&nbsp;{flag}", unsafe_allow_html=True)

        elif analyze_btn:
            st.warning("Please enter a message first.")
        else:
            st.markdown(
                """
<div style="background:#16213e;border:1px dashed #0f3460;border-radius:12px;padding:2rem;text-align:center;color:#a8b2d8;">
  <p style="font-size:2rem;">🔍</p>
  <p>Enter a message and click <b>Analyze</b> to see the result here.</p>
</div>""",
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 📈 Model Performance Dashboard")

    r1c1, r1c2 = st.columns(2)

    # Confusion matrix
    with r1c1:
        st.markdown("#### Confusion Matrix")
        fig_cm = go.Figure(
            go.Heatmap(
                z=cm,
                x=["Predicted HAM", "Predicted SPAM"],
                y=["Actual HAM", "Actual SPAM"],
                colorscale=[[0, "#16213e"], [0.5, "#0f3460"], [1, "#e94560"]],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 22, "color": "white"},
                showscale=False,
            )
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1a1a2e",
            font_color="#a8b2d8",
            height=320,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(tickfont=dict(color="#a8b2d8")),
            yaxis=dict(tickfont=dict(color="#a8b2d8")),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve
    with r1c2:
        st.markdown(f"#### ROC Curve  (AUC = {roc_auc:.3f})")
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.3f}",
                       line=dict(color="#e94560", width=3))
        )
        fig_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                       line=dict(color="#a8b2d8", dash="dash", width=1))
        )
        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1a1a2e",
            font_color="#a8b2d8", height=320,
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    # Precision-Recall
    with r2c1:
        st.markdown("#### Precision-Recall Curve")
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(x=rec, y=prec, mode="lines", name="PR Curve",
                       line=dict(color="#2ecc71", width=3), fill="tozeroy",
                       fillcolor="rgba(46,204,113,0.1)")
        )
        fig_pr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1a1a2e",
            font_color="#a8b2d8", height=300,
            xaxis_title="Recall", yaxis_title="Precision",
            margin=dict(t=20, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # Per-class metrics bar chart
    with r2c2:
        st.markdown("#### Per-Class Metrics")
        classes = ["HAM (0)", "SPAM (1)"]
        metrics_names = ["Precision", "Recall", "F1-Score"]
        colors = ["#2ecc71", "#e94560", "#3498db"]
        fig_bar = go.Figure()
        for i, (mn, col) in enumerate(zip(metrics_names, colors)):
            vals = [report[str(k)][mn.lower().replace("-", "-")] for k in [0, 1]]
            fig_bar.add_trace(go.Bar(name=mn, x=classes, y=vals,
                                     marker_color=col, opacity=0.85))
        fig_bar.update_layout(
            barmode="group", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1a1a2e", font_color="#a8b2d8", height=300,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(range=[0, 1.1]),
            margin=dict(t=20, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Dataset distribution
    st.markdown("#### 📂 Training Dataset Distribution")
    label_counts = df["label"].value_counts()
    fig_pie = go.Figure(
        go.Pie(
            labels=["HAM", "SPAM"],
            values=[label_counts[0], label_counts[1]],
            hole=0.55,
            marker=dict(colors=["#2ecc71", "#e94560"],
                        line=dict(color="#1a1a2e", width=3)),
            textfont=dict(color="white", size=14),
        )
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#a8b2d8",
        height=300, margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text=f"{len(df)}<br>msgs", x=0.5, y=0.5,
                         font_size=18, font_color="#e8e8e8", showarrow=False)],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 3 — BATCH TESTER
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 🧪 Batch Message Tester")
    st.markdown("<p style='color:#a8b2d8;'>Enter one message per line and analyze them all at once.</p>", unsafe_allow_html=True)

    default_batch = "\n".join([
        "Hey, are we still meeting today?",
        "Win money now!!! Click http://free-money-login.xyz",
        "Check this out https://recruitment.cdcfib.gov.ng/",
        "URGENT! Verify your account https://secure-bank-update-login.xyz",
        "Looking forward to seeing you this weekend!",
        "FREE! Claim your prize at http://bonus-win-free.xyz/verify",
    ])

    batch_input = st.text_area(
        "Messages (one per line):", value=default_batch, height=200, key="batch"
    )

    if st.button("🚀 Analyze All Messages", use_container_width=True, key="batch_btn"):
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("Please enter at least one message.")
        else:
            results = []
            prog = st.progress(0, text="Analyzing messages…")
            for i, msg in enumerate(lines):
                verdict, text_pred, proba, url, risk = final_prediction(msg)
                results.append({
                    "Message": msg[:70] + ("…" if len(msg) > 70 else ""),
                    "Verdict": verdict,
                    "Spam Prob %": f"{proba*100:.1f}%",
                    "URL": url if url else "None",
                    "URL Risk": risk if url else "—",
                })
                prog.progress((i + 1) / len(lines), text=f"Analyzing {i+1}/{len(lines)}…")
            prog.empty()

            results_df = pd.DataFrame(results)

            # Summary
            n_spam = sum(1 for r in results if r["Verdict"] == "SPAM")
            n_ham = len(results) - n_spam
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Messages", len(results))
            s2.metric("🚨 SPAM", n_spam)
            s3.metric("✅ HAM", n_ham)

            # Styled table
            def color_row(row):
                color = "background-color:#2a1a1a; color:#e94560" if row["Verdict"] == "SPAM" \
                        else "background-color:#1a2a1a; color:#2ecc71"
                return [color] * len(row)

            styled = results_df.style.apply(color_row, axis=1).set_properties(
                **{"text-align": "left"}
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
