import streamlit as st
import joblib
import re
import numpy as np
from urllib.parse import urlparse

# Page configuration
st.set_page_config(
    page_title="Hybrid Spam Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .spam-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .ham-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #f44336;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .risk-low {
        color: #4caf50;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_models():
    try:
        text_model = joblib.load('text_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return text_model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found! Please ensure 'text_model.pkl' and 'vectorizer.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

# Helper functions (consistent with training)
def extract_url_from_text(text):
    """Extract first URL from text"""
    urls = re.findall(r'https?://\S+', text)
    return urls[0] if urls else None

def url_risk_score(url):
    """Calculate risk score for a URL (0 = safe, higher = riskier)"""
    score = 0
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Trusted domains - return 0 immediately
    trusted_domains = [
        'google.com', 'github.com', 'microsoft.com',
        'openai.com', 'wikipedia.org', 'linkedin.com',
        'amazon.com', 'apple.com', 'netflix.com'
    ]
    
    if any(td in domain for td in trusted_domains):
        return 0
    
    # Suspicious keywords
    suspicious_words = [
        'login', 'verify', 'secure', 'bank', 'update',
        'free', 'bonus', 'win', 'prize', 'account',
        'confirm', 'alert', 'security', 'validate'
    ]
    
    # Risk indicators
    if len(url) > 75:
        score += 1
    if '@' in url or '%' in url:
        score += 2
    if url.count('-') > 2:
        score += 1
    if not url.startswith('https'):
        score += 1
    if domain.count('.') > 2:
        score += 1
    
    for word in suspicious_words:
        if word in url.lower():
            score += 1
    
    return min(score, 10)  # Cap at 10

def final_prediction(message, text_model, vectorizer):
    """
    Hybrid prediction combining text classification and URL risk analysis.
    Note: Uses original message (including URLs) for text classification
    to maintain consistency with training data.
    """
    # Extract URL if present
    url = extract_url_from_text(message)
    
    # Text classification on ORIGINAL message (matching training)
    vectorized = vectorizer.transform([message])
    text_pred = text_model.predict(vectorized)[0]
    text_proba = text_model.predict_proba(vectorized)[0]
    spam_probability = text_proba[1]  # Probability of spam class
    
    # URL risk analysis
    risk_score = 0
    risk_details = {}
    if url:
        risk_score = url_risk_score(url)
        risk_details = {
            'url': url,
            'risk_score': risk_score,
            'domain': urlparse(url).netloc
        }
    
    # Final logic
    if text_pred == 1:
        final_result = "SPAM"
        reason = "Text content classified as spam"
    elif risk_score >= 3:
        final_result = "SPAM"
        reason = f"Suspicious URL detected (risk score: {risk_score}/10)"
    else:
        final_result = "NOT SPAM"
        reason = "Safe content and URL"
    
    return {
        'final': final_result,
        'reason': reason,
        'text_prediction': 'Spam' if text_pred == 1 else 'Not Spam',
        'spam_probability': spam_probability,
        'has_url': url is not None,
        'url_risk_score': risk_score,
        'url_details': risk_details
    }

# Main application
def main():
    # Header
    st.title("🛡️ Hybrid Spam Detection System")
    st.markdown("""
    <p style='font-size: 1.1rem; color: #666;'>
    Advanced spam detection combining <b>machine learning</b> (TF-IDF + Logistic Regression) 
    with <b>URL risk analysis</b> for comprehensive protection.
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Performance")
        st.markdown("""
        ### Classification Metrics
        | Class | Precision | Recall | F1-Score |
        |-------|-----------|--------|----------|
        | Ham (0) | 0.98 | 0.99 | 0.99 |
        | Spam (1) | 0.94 | 0.90 | 0.92 |
        
        **Overall Accuracy:** 98%
        """)
        
        st.markdown("---")
        st.header("🎯 How It Works")
        st.markdown("""
        1. **Text Analysis**: TF-IDF features + Logistic Regression
        2. **URL Analysis**: Risk scoring based on:
           - Domain reputation
           - URL length & structure
           - Suspicious keywords
           - HTTPS usage
        3. **Hybrid Decision**: Combines both signals for final verdict
        """)
        
        st.markdown("---")
        st.header("⚠️ URL Risk Indicators")
        st.markdown("""
        - ❌ Untrusted domain
        - 🔗 Long URL (>75 chars)
        - 🔒 Missing HTTPS
        - 📧 Contains '@' or '%'
        - 🚩 Suspicious keywords (login, verify, etc.)
        """)
    
    # Main content area - two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Message Analysis")
        # Text input
        user_input = st.text_area(
            "Enter the message to analyze:",
            height=150,
            placeholder="Paste or type an email, SMS, or chat message here...",
            key="message_input"
        )
        
        # Example messages
        st.markdown("**Quick Examples:**")
        example_col1, example_col2 = st.columns(2)
        
        examples = {
            "Safe Message": "Hey, are we still meeting at the cafe at 3pm?",
            "Spam Message": "WINNER! You've won $1000 gift card! Click http://prize-claim-now.xyz",
            "Phishing URL": "URGENT: Your account will be suspended. Verify now: http://secure-login-alert.com",
            "Neutral with URL": "Check out this cool article https://github.com/features"
        }
        
        with example_col1:
            if st.button("📧 Safe Message", use_container_width=True):
                user_input = examples["Safe Message"]
                st.session_state.message_input = user_input
                st.rerun()
            if st.button("🎁 Spam Message", use_container_width=True):
                user_input = examples["Spam Message"]
                st.session_state.message_input = user_input
                st.rerun()
        
        with example_col2:
            if st.button("⚠️ Phishing URL", use_container_width=True):
                user_input = examples["Phishing URL"]
                st.session_state.message_input = user_input
                st.rerun()
            if st.button("🔗 Neutral URL", use_container_width=True):
                user_input = examples["Neutral with URL"]
                st.session_state.message_input = user_input
                st.rerun()
    
    with col2:
        st.subheader("📈 Analysis Stats")
        st.markdown("""
        <div class="metric-card">
            <h3>📊 98%</h3>
            <p>Model Accuracy</p>
        </div>
        <div class="metric-card">
            <h3>🔍 5000</h3>
            <p>TF-IDF Features</p>
        </div>
        <div class="metric-card">
            <h3>⚖️ Balanced</h3>
            <p>Class Weights</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analyze button
    st.markdown("---")
    
    if st.button("🔍 Analyze Message", type="primary", use_container_width=True):
        if not user_input or user_input.strip() == "":
            st.warning("⚠️ Please enter a message to analyze.")
        else:
            # Load models
            text_model, vectorizer = load_models()
            
            # Get prediction
            with st.spinner("Analyzing message..."):
                result = final_prediction(user_input, text_model, vectorizer)
            
            # Display results
            st.markdown("### 📋 Analysis Results")
            
            # Main verdict with colored box
            if result['final'] == "SPAM":
                st.markdown(f"""
                <div class="spam-box">
                    <h2 style="color: #f44336; margin: 0;">⚠️ SPAM DETECTED</h2>
                    <p style="margin-top: 0.5rem;"><b>Reason:</b> {result['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ham-box">
                    <h2 style="color: #4caf50; margin: 0;">✅ NOT SPAM</h2>
                    <p style="margin-top: 0.5rem;"><b>Reason:</b> {result['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                text_color = "🟢" if result['text_prediction'] == "Not Spam" else "🔴"
                st.metric(
                    "Text Classification",
                    f"{text_color} {result['text_prediction']}",
                    delta=f"Confidence: {result['spam_probability']:.1%}" if result['text_prediction'] == "Spam" else None
                )
            
            with col2:
                url_status = "✅ No URL" if not result['has_url'] else f"🔗 URL Found (Risk: {result['url_risk_score']}/10)"
                risk_class = ""
                if result['has_url']:
                    if result['url_risk_score'] >= 3:
                        risk_class = "risk-high"
                    elif result['url_risk_score'] >= 1:
                        risk_class = "risk-medium"
                    else:
                        risk_class = "risk-low"
                st.metric(
                    "URL Analysis",
                    url_status,
                    delta=f"Score: {result['url_risk_score']}/10" if result['has_url'] else None,
                    delta_color="inverse" if result['has_url'] and result['url_risk_score'] >= 3 else "normal"
                )
            
            with col3:
                st.metric("Final Verdict", result['final'])
            
            # Show detailed URL info if present
            if result['has_url'] and result['url_details']:
                with st.expander("🔗 URL Details"):
                    st.markdown(f"""
                    - **URL:** `{result['url_details']['url']}`
                    - **Domain:** `{result['url_details']['domain']}`
                    - **Risk Score:** {result['url_details']['risk_score']}/10
                    """)
                    
                    # Risk breakdown explanation
                    st.markdown("**Risk Factors:**")
                    url = result['url_details']['url']
                    risks = []
                    if len(url) > 75:
                        risks.append("• Very long URL (>75 chars)")
                    if '@' in url or '%' in url:
                        risks.append("• Contains suspicious characters (@ or %)")
                    if url.count('-') > 2:
                        risks.append("• Multiple hyphens in URL")
                    if not url.startswith('https'):
                        risks.append("• Does not use HTTPS (insecure)")
                    if urlparse(url).netloc.count('.') > 2:
                        risks.append("• Multiple subdomains (possible typosquatting)")
                    
                    if risks:
                        for risk in risks:
                            st.markdown(risk)
                    else:
                        st.markdown("• No specific risk indicators detected")
            
            # Show spam probability gauge
            st.markdown("### 📊 Spam Probability")
            prob_percentage = result['spam_probability'] * 100
            st.progress(prob_percentage / 100)
            st.caption(f"Model confidence: {prob_percentage:.1f}% that this is spam")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem;">
        🔒 Hybrid detection system | Machine Learning + URL Risk Analysis | High accuracy spam filtering
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()